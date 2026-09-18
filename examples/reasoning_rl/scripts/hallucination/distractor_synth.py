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
"""D17 distractor synthesis: *answerable, but with a red herring* rows.

Design doc sections 4.9.4 (table C) and D17.  GSM-IC's ``sentence_template``
column is a rule-based distractor engine: each template is parameterised only by
``{role}`` and ``{number}``, so rendering it with any pair yields a well-formed
extra sentence.  Attaching such a sentence to an *answerable* base problem leaves
the answer untouched -- the sentence states an irrelevant fact -- which is what
makes the required contrast pair work:

    distractor (one irrelevant condition added, still answerable)
        vs  SUM-del / TreeCut / UMWP-cat1 / MiP (one necessary premise removed,
            unanswerable)

The recon measured the template inventory at **394** distinct templates over
GSM-IC's 58,052 rows, not the 242 the design doc quotes (242 is the 2step-only
count).  Replay is exact: 58,052/58,052 rows reproduce by plain substitution.

Two things the naive "just render every template" version gets wrong, both of
which this module refuses to do:

1. **Not every template is answer-preserving.**  A comparative or additive
   sentence *does* add a constraint (``{role} is {number} inches taller than
   Steve.``), and one that mentions a hard-coded third-party name can collide
   with the base problem's own actor.  394 templates minus those leaves the
   engine's usable inventory, which :func:`load_safe_templates` reports by
   rejection reason rather than silently dropping.

2. **An overlapped role must not restate a known property.**  When the
   distractor names someone already in the base problem, the sentence has to
   introduce a *new* property or it contradicts the premise ("Steve is 5'6\"" +
   "The height of Steve is 8 feet.").  The guard is conservative: for an
   overlapped role the sentence may not reuse any content word of the base
   problem, which structurally makes ``overlapped`` and ``in_topic`` mutually
   exclusive.  Six of the eight label cells are therefore reachable, and the
   allocator reports which ones it could not fill instead of inventing rows.

Row shape is the usual :mod:`schema` contract: template B (answer or refuse),
gold = the base problem's original answer, so the reward path is the existing
``_math_score`` and no new scoring logic is introduced (design doc section
4.9.4: "答案与 reward 完全不变").

Standing design-doc deviation, recorded here and surfaced in the build report:

* Table B row 1 ("solvable numeric + distractor") is specified as
  ``GSM-IC 2,000 + synthesised 2,400``, and section 4.9.4 splits those 2,400 as
  SUM 1,000 / UMWP 600 / K&K 400 / main pool 400.  A K&K base answers with a
  *role sequence*, not a number, so its 400 rows cannot carry the
  ``solvable_numeric`` branch label.  They are emitted under
  :data:`POOL_BRANCH` as ``solvable_roles`` instead; the row total is unchanged
  and only the branch bookkeeping moves.

Usage::

    /lhy/miniconda3/envs/lhy/bin/python distractor_synth.py --selftest
    /lhy/miniconda3/envs/lhy/bin/python distractor_synth.py

Rows land in :data:`DEFAULT_OUT`, next to every other adapter's output, because
``mix_halluc.py`` reads exactly one directory and the synthesised slice is one of
the quota cells it must find there (table B row 1: ``(solvable_numeric, SUM)``
and friends).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import schema

DEFAULT_RAW_DIR = os.path.expanduser("~/data/reasoning_rl/halluc/raw")
#: Where the built rows and the build report go.  Same directory as the other
#: adapters' ``DEFAULT_OUT`` -- ``mix_halluc.py`` reads one directory, and the
#: synthesised slice *is* the ``(solvable_numeric, SUM/UMWP/KK/MAIN)`` cells.
DEFAULT_OUT = os.path.expanduser("~/data/reasoning_rl/halluc/built/d17.parquet")
GSMIC_FILES = ("GSM-IC_2step.json", "GSM-IC_mstep.json")


def report_path_for(out: str) -> str:
    """The report that belongs to ``--out``: same stem, ``_report.json``.

    Derived from ``--out`` instead of pinned to the build directory, so that a
    scratch build (``--out /tmp/d17.parquet``) cannot overwrite the canonical
    ``d17_report.json`` -- the file the build report cites -- while the rows it
    does not touch stay in place.  Same convention in every adapter here.
    """
    return os.path.splitext(out)[0] + "_report.json"

# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------

POOL_SUM = "sum"
POOL_UMWP = "umwp"
POOL_KK = "kk"
POOL_MAIN = "main"
POOLS = (POOL_SUM, POOL_UMWP, POOL_KK, POOL_MAIN)

# Design doc section 4.9.4: SUM-answerable 1,000 / UMWP-answerable 600 / K&K 400
# / main pool 400.
DEFAULT_POOL_QUOTA: dict[str, int] = {
    POOL_SUM: 1000,
    POOL_UMWP: 600,
    POOL_KK: 400,
    POOL_MAIN: 400,
}

POOL_DATA_SOURCE: dict[str, str] = {
    POOL_SUM: schema.SOURCE_SUM,
    POOL_UMWP: schema.SOURCE_UMWP,
    POOL_KK: schema.SOURCE_KK,
    POOL_MAIN: schema.SOURCE_MAIN,
}

POOL_BRANCH: dict[str, str] = {
    POOL_SUM: schema.BRANCH_SOLVABLE_NUMERIC,
    POOL_UMWP: schema.BRANCH_SOLVABLE_NUMERIC,
    # See the module docstring: a K&K base answers with role words, so these rows
    # ride the roles branch rather than the numeric one.
    POOL_KK: schema.BRANCH_SOLVABLE_ROLES,
    POOL_MAIN: schema.BRANCH_SOLVABLE_NUMERIC,
}

POOL_RAW_PATH: dict[str, str] = {
    POOL_SUM: "sum/train.parquet",
    POOL_UMWP: "umwp/StandardDataset.jsonl",
    POOL_KK: "kk",
}

# The perturbation vocabulary entry these rows use (design doc section 3).
DISTRACTOR_PERTURBATION = "distracting_condition"

# ---------------------------------------------------------------------------
# the three label axes and their balance targets
# ---------------------------------------------------------------------------

AXIS_TARGETS: dict[str, dict[str, float]] = {
    "role_label": {"overlapped": 0.50, "nonoverlapped": 0.50},
    "number_label": {"in_range": 0.50, "out_range": 0.50},
    # Design doc table C: GSM-IC's own in/out split is 15,404 : 18,816 = 45:55.
    "sentence_label": {"in_topic": 0.45, "out_topic": 0.55},
}
AXES = ("role_label", "number_label", "sentence_label")
AXIS_TOLERANCE = 0.05

# ---------------------------------------------------------------------------
# template safety
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{(?:role|number)\}")

# A comparison or an aggregation adds a *constraint*, so the sentence stops being
# a red herring and starts changing what the problem asks.
RELATIONAL_MARKERS = (
    # The bare " than " catches the whole comparative family, including the ones
    # with a noun in the middle: "{role} has {number} more potatoes than Oli."
    # Matching only "more than" misses that one entirely.
    " than ",
    "more than",
    "less than",
    "fewer than",
    "taller than",
    "shorter than",
    "older than",
    "younger than",
    "heavier than",
    "lighter than",
    "longer than",
    "wider than",
    "deeper than",
    "greater than",
    "bigger than",
    "larger than",
    "smaller than",
    "higher than",
    "lower than",
    "faster than",
    "slower than",
    "times as many",
    "times more",
    "times less",
    "as many as",
    "as much as",
    "twice as",
    "half as",
)
# Markers that make a sentence aggregate over the *whole* problem, so the gold
# answer would stop being well-defined.  "in addition" is deliberately absent: it
# is a discourse conjunction, not an aggregation, and the seven templates that
# open with it (488 of 58,052 upstream rows) stay safe because the sentence still
# states a standalone fact about a different actor.
ADDITIVE_MARKERS = (
    "in total",
    "altogether",
    "combined",
    "together",
    "in all",
)
# A truth-functional sentence about a K&K inhabitant would change the puzzle.
TRUTH_FUNCTIONAL_MARKERS = (
    "knight",
    "knave",
    "liar",
    "lies",
    "lying",
    "truth",
    "truthful",
    "assert",
    "claim",
    "says",
    "said",
    "tells",
    "honest",
)

# Capitalised tokens that are legitimately not proper names of a *third party*.
# The source really does contain the misspelling "Feburary".
SAFE_PROPER = frozenset(
    """
    January February Feburary March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday
    The A An In On At For From By With I
    """.split()
)

# Capitalised tokens that are verbs, quantifiers or discourse markers rather than
# names -- the vocabulary that made an earlier revision emit "Create ate 4 pounds
# of chocolate." and "Proposed fed n/a monkeys." out of MATH problems.
#
# This is applied at *every* position, not only at the start of a sentence: a
# capitalised "Additionally" or "Equivalently" also follows a comma, a colon or a
# semicolon, which is where the sentence splitter cannot help.  Lower-case
# lookalikes are already excluded by the capitalisation test in the miner.
NON_NAME_OPENERS = frozenset(
    """
    determine compute calculate find solve evaluate simplify suppose given let assume
    consider define prove express note recall show write answer round use using
    additionally however therefore thus also now first second third finally next then
    hence since because although though while when after before during every another
    one two three four five six seven eight nine ten half given assuming proposed based
    according among inside another circle triangle square rectangle point points line
    school cartesian elementary ferris equivalently similarly conversely respectively
    enter please estimate convert draw label plot graph compare list describe explain
    state verify check test apply substitute multiply divide add subtract derive
    obtain yield resulting result follows note that observe henceforth
    """.split()
)

# Ordinary proper nouns that a word problem may capitalise but that cannot act as
# the subject of "{role} bought {number} newspapers.": titles, places,
# nationalities, holidays, teams.  Every one of these was measured in the mined
# pool before being listed.
NON_ACTOR_PROPER = frozenset(
    """
    Mr Mrs Ms Dr Miss Professor Doctor Aunt Uncle Sir Lady
    China Chinese America American England English Britain British France French
    Spain Spanish Germany German Japan Japanese Korea Korean India Indian Canada
    Canadian Texas Arkansas Nebraska Shandong Australia Russia Russian Mexico
    Halloween Christmas Easter Thanksgiving Facebook Razorback Bonanza
    """.split()
)

# Words that every GSM-IC template and every word problem share, and which
# therefore say nothing about whether the two are *about* the same thing.
# "{role} bought {number} newspapers." must not count as in-topic against "Doug
# bought 8 panes of glass" just because both say "bought" -- matching on the
# generic verb would make the in/out gradient meaningless, which is worse than
# having a small in-topic supply.
GENERIC_TOPIC_WORDS = frozenset(
    """
    bought buys buy boughts has have had having got gets got sold sells sell earns earn
    read reads rides ride raised raises raise gave gives give made makes make spends spend
    costs cost pays pay picked pick found find collected collect received receive ordered
    order planted plant ate eat drank drink watched watch studied study took take takes
    day days week weeks month months year years hour hours minute minutes second seconds
    past ago every each next last today yesterday tomorrow morning night time times
    grocery store school student students class classes people person new old more less
    """.split()
)


def hardcoded_names(template: str) -> list[str]:
    """Proper-looking tokens baked into a template (``Steve``, ``Oli``, ...).

    Skipped: the sentence-initial capital (``The height of {role} ...``) and the
    small :data:`SAFE_PROPER` allowlist of months, days and articles.  Anything
    left is a name the template hard-codes, i.e. a comparison anchor we did not
    choose and cannot substitute.
    """
    names: list[str] = []
    for position, token in enumerate(template.split()):
        word = token.strip(".,;:!?\"'()$")
        if not word or not word[:1].isupper() or word.isupper():
            continue
        if position == 0 and not _PLACEHOLDER_RE.match(token):
            continue
        if word in SAFE_PROPER:
            continue
        names.append(word)
    return names


def unsafe_reason(template: str) -> str | None:
    """Why a template may not be used, or ``None`` when it is safe.

    Returns one of ``empty``, ``no_placeholder``, ``single_placeholder``,
    ``question_form``, ``relational``, ``additive``, ``truth_functional``,
    ``hardcoded_name``.
    """
    text = (template or "").strip()
    if not text:
        return "empty"
    if not _PLACEHOLDER_RE.search(text):
        return "no_placeholder"
    if "{role}" not in text or "{number}" not in text:
        # Both axes have to be carried by the sentence for its labels to mean
        # anything: a sentence with no actor cannot be overlapped or
        # nonoverlapped, and one with no number cannot be in_range or out_range.
        # 50 of the 394 distinct templates are number-only and 1 is role-only.
        return "single_placeholder"
    if "?" in text:
        return "question_form"
    lowered = text.casefold()
    for marker in RELATIONAL_MARKERS:
        if marker in lowered:
            return "relational"
    for marker in ADDITIVE_MARKERS:
        if marker in lowered:
            return "additive"
    for marker in TRUTH_FUNCTIONAL_MARKERS:
        if marker in lowered:
            return "truth_functional"
    if hardcoded_names(text):
        return "hardcoded_name"
    return None


def template_words(template: str, *, topic_only: bool = False) -> set[str]:
    """Content words of a rendered-away template (no role, no number).

    ``topic_only`` drops :data:`GENERIC_TOPIC_WORDS`, which is what the in/out
    *label* uses; the overlap guard uses the full set, because for a guard an
    over-eager match is the safe direction.
    """
    stripped = _PLACEHOLDER_RE.sub(" ", template)
    words = {
        token.casefold()
        for token in schema.words(stripped)
        if schema.is_content_word(token) and not token.isdigit()
    }
    if topic_only:
        words -= GENERIC_TOPIC_WORDS
    return words


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One distinct GSM-IC ``sentence_template`` with a representative fill."""

    sentence_template: str
    role: str
    number: str
    role_label: str
    number_label: str
    sentence_label: str
    count: int = 0

    @property
    def content_words(self) -> set[str]:
        """All content words -- the contradiction guard's vocabulary."""
        return template_words(self.sentence_template)

    @property
    def topic_words(self) -> set[str]:
        """Content words minus generic verbs -- the topicality axis' vocabulary."""
        return template_words(self.sentence_template, topic_only=True)

    def render(self, role: str, number: str) -> str:
        """Substitute exactly the way the recon replayed all 58,052 rows."""
        return (
            self.sentence_template.replace("{role}", role)
            .replace("{number}", number)
            .strip()
        )


def load_templates(raw_dir: str | os.PathLike = DEFAULT_RAW_DIR) -> list[Template]:
    """Read both GSM-IC files and return one :class:`Template` per distinct string.

    The representative fill comes from the most common ``(role, number, labels)``
    quadruple observed for that template, so a caller that ignores the labels and
    just renders ``role``/``number`` still gets a source-faithful sentence.
    """
    gsmic_dir = Path(raw_dir) / "gsmic"
    buckets: dict[str, Counter] = defaultdict(Counter)
    for name in GSMIC_FILES:
        path = gsmic_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing -- run fetch_raw.py --only gsmic first"
            )
        with path.open(encoding="utf-8") as handle:
            for row in json.load(handle):
                template = (row.get("sentence_template") or "").strip()
                if not template:
                    continue
                key = (
                    row.get("role", ""),
                    row.get("number", ""),
                    row.get("role_label", ""),
                    row.get("number_label", ""),
                    row.get("sentence_label", ""),
                )
                buckets[template][key] += 1

    templates: list[Template] = []
    for template, counter in buckets.items():
        (role, number, role_label, number_label, sentence_label), count = (
            counter.most_common(1)[0]
        )
        templates.append(
            Template(
                sentence_template=template,
                role=role,
                number=number,
                role_label=role_label,
                number_label=number_label,
                sentence_label=sentence_label,
                count=sum(counter.values()),
            )
        )
    templates.sort(key=lambda t: (-t.count, t.sentence_template))
    return templates


def load_safe_templates(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR,
) -> tuple[list[Template], dict[str, Any]]:
    """Templates minus the ones that could change the answer, plus a funnel.

    Returns ``(safe, report)`` where ``report`` counts every distinct template
    once, by rejection reason, so a caller can state how much of the inventory it
    used rather than reporting a silent cap.
    """
    all_templates = load_templates(raw_dir)
    reasons: Counter = Counter()
    safe: list[Template] = []
    rejected: list[dict[str, str]] = []
    for template in all_templates:
        reason = unsafe_reason(template.sentence_template)
        if reason is None:
            safe.append(template)
        else:
            reasons[reason] += 1
            rejected.append(
                {"template": template.sentence_template, "reason": reason}
            )
    report = {
        "distinct_total": len(all_templates),
        "safe": len(safe),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "rows_covered_by_safe": sum(t.count for t in safe),
        "rows_total": sum(t.count for t in all_templates),
        "rejected_examples": rejected[:25],
    }
    return safe, report


# ---------------------------------------------------------------------------
# base problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseQuestion:
    """One answerable problem the distractor sentence gets attached to."""

    question: str
    answer: str
    pool: str
    # A pool-unique, upstream-traceable key, and the only identifier allowed into a
    # ``task_id``.  ``index`` alone is not enough: K&K's five per-size parquet files
    # each number their rows 0..999, so indices repeat five times over within the
    # pool (UMWP's ``id`` and SUM's row ordinal happen to be unique today, but that
    # is upstream's business, not a property this module should assume).
    uid: str = ""
    split: str = "train"
    index: int = -1
    # K&K only: the row's own [truth-teller word, liar word], so the synthesised
    # row keeps the same answer-mapping contract as its unsynthesised siblings.
    role_words: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


def _float_numbers(text: str) -> list[float]:
    out = []
    for token in schema.numbers_in(text):
        try:
            out.append(float(token))
        except ValueError:  # pragma: no cover - numbers_in only yields literals
            continue
    return out


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def base_names(text: str) -> list[str]:
    """The base problem's own actor names, in document order, for overlapped fills.

    Two rules, because neither alone is enough:

    * A capitalised content word anywhere may be a name (``Jewel``, ``Doug``), but
      months, days and stop words are not.
    * A **sentence-initial** capital may still be a name -- UMWP problems open on
      their actor, "Bryan took a look at his books and magazines." -- so it cannot
      simply be skipped.  It is accepted unless it is a known non-name opener,
      which is what keeps "Determine [(1 (x) 2) (x) 3]" from offering a role
      called ``Determine``.

    Sentences are split from the raw text, not from :func:`schema.words`: that
    helper drops punctuation, so "b = a/b. Determine ..." looks like one sentence
    and every verb after a full stop reads as mid-sentence.

    The result is ordered by position, and :func:`choose_distractor` prefers the
    earliest entries because in a word problem the actor is usually named in the
    first few words.  It is a heuristic, not a name recogniser: a proper noun that
    is not a person ("At Euclid Middle School ..." yields ``Euclid``/``Middle``/
    ``School``) can still be picked, and the row stays a valid -- if odd-sounding
    -- distractor row.  The build report records how often that happens.
    """
    names: list[str] = []
    for sentence in _SENTENCE_RE.findall(text):
        for position, token in enumerate(schema.words(sentence)):
            word = token.strip(".,;:!?\"'()$")
            if not word or not word[:1].isupper() or word.isupper():
                continue
            if word in SAFE_PROPER or not schema.is_content_word(word):
                continue
            if position == 0 and word.casefold() in NON_NAME_OPENERS:
                continue
            names.append(word)
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# actor mining
# ---------------------------------------------------------------------------
#
# The *overlapped* role axis needs an actor the base problem already names, so it
# has to be mined out of the problem text.  Mining capitalised words naively is
# how an earlier revision produced "Create ate 4 pounds of chocolate." and
# "Proposed fed n/a monkeys.": MATH problems open sentences with imperatives
# ("Create a five-digit number...", "Proposed ..."), and a capitalised verb is
# indistinguishable from a capitalised name without knowing what a name is.
#
# What a name is here is *bootstrapped from the two elementary pools*, whose word
# problems are about people: a candidate must (a) be Titlecase, (b) not occur in
# lower case anywhere in the corpus -- which is what separates "Point" from
# "Alice", since the corpus does say "point" -- (c) not be a known verb or
# discourse marker, and (d) be attested as an actor at least
# ACTOR_MIN_POOL_SUPPORT times in UMWP or be one of K&K's inhabitants.
#
# The cost is supply, and it is measured rather than wished away: most MATH
# problems have no actor at all, so the SUM pool can only fill the overlapped
# half of its role quota on ~1 base in 10.  ``synthesise`` therefore plans the
# role axis against a *global* budget it can actually meet and records the
# shortfall against the design target in its report.

_NON_ACTOR_PROPER_LOWER = frozenset(name.casefold() for name in NON_ACTOR_PROPER)
# A possessive clitic on the end of a mined token.  It has to come *off* rather
# than disqualify the token: the base writes "Bill's brother has 5 apples", the
# actor the sentence may reuse is Bill, and substituting the token as it stands
# renders "Bill's baked 12 pieces of breads." -- or, into a template that carries
# its own clitic, "John's's monthly rent is $10000."  Measured before the strip:
# 87 of the SUM pool's 1,299 actor-bearing bases mined a possessive, of which 25
# reached the shipped 2,000 rows.
_POSSESSIVE_RE = re.compile(r"['’]s?$")

ACTOR_MIN_POOL_SUPPORT = 3
# The pools whose questions are about people, and which therefore define the
# person-name vocabulary.  SUM is deliberately not among them -- it is the pool
# being filtered, and a vocabulary trained on it would inherit its noise.
ACTOR_VOCABULARY_POOLS = (POOL_UMWP, POOL_KK)


def lowercase_vocabulary(pools: Mapping[str, Sequence["BaseQuestion"]]) -> set[str]:
    """Every content word that occurs in lower case anywhere in the pools.

    The one signal that reliably separates a proper noun from a sentence-initial
    verb in this corpus: the problems write "Alice" and also write "point", but
    never write "alice".
    """
    vocabulary: set[str] = set()
    for bases in pools.values():
        for base in bases:
            for token in schema.words(base.question):
                if token[:1].islower() and schema.is_content_word(token):
                    vocabulary.add(token.casefold())
    return vocabulary


def actor_candidates(text: str, lowercase: Collection[str]) -> list[str]:
    """Capitalised tokens of ``text`` that could be the subject of a template.

    A possessive is reduced to its stem rather than dropped (:data:`_POSSESSIVE_RE`),
    so ``"Bill's"`` yields the actor ``Bill`` -- which is still a name the base
    carries, and one a template can take.
    """
    candidates: list[str] = []
    for token in base_names(text):
        name = _POSSESSIVE_RE.sub("", token)
        if not (name[:1].isupper() and name[1:].islower()):
            # Titlecase only: "EndArrow" and "A--B--C--cycle" are LaTeX macro
            # names, and a single letter is an article, not an actor.
            continue
        key = name.casefold()
        if (
            key in lowercase
            or key in NON_NAME_OPENERS
            # Both tables are matched case-insensitively: the openers are written
            # lower case, the proper nouns are written as they appear in the text,
            # and a capitalised "Chinese" that slipped through here would be
            # rendered as an actor.
            or key in _NON_ACTOR_PROPER_LOWER
        ):
            continue
        if not schema.is_content_word(name) or name in SAFE_PROPER:
            continue
        candidates.append(name)
    return candidates


def mine_actors(
    pools: Mapping[str, Sequence[BaseQuestion]],
    *,
    min_support: int = ACTOR_MIN_POOL_SUPPORT,
) -> dict[str, Any]:
    """Fill ``base.meta["names"]`` with the actors each base may be overlapped on.

    Runs *after* the pools are loaded and *before* any candidate is built, because
    both the overlapped role pool and the sampler's inverted index read
    ``meta["names"]``.  A base with an empty list simply has no overlapped cell --
    which is the fail-closed default: forgetting to mine yields nonoverlapped rows,
    never a fabricated actor.

    Returns the report the build records, so the supply this rule costs is visible
    in the artefacts rather than only in the marginals it moved.
    """
    lowercase = lowercase_vocabulary(pools)

    support: Counter[str] = Counter()
    for pool in ACTOR_VOCABULARY_POOLS:
        for base in pools.get(pool) or []:
            support.update(set(actor_candidates(base.question, lowercase)))
    vocabulary = {name for name, count in support.items() if count >= min_support}
    # K&K's inhabitants are already the ground truth for that pool -- the puzzle
    # lists them -- so they seed the vocabulary rather than being filtered by it.
    for base in pools.get(POOL_KK) or []:
        vocabulary.update(base.meta.get("names") or [])
    lowered_vocabulary = {name.casefold() for name in vocabulary}

    report: dict[str, Any] = {
        "lowercase_vocabulary": len(lowercase),
        "vocabulary": len(vocabulary),
        "min_pool_support": min_support,
        "pools": {},
    }
    for pool, bases in pools.items():
        with_actor = 0
        distinct: Counter[str] = Counter()
        for base in bases:
            if pool == POOL_KK:
                names = list(base.meta.get("names") or [])
            else:
                names = [
                    name
                    for name in actor_candidates(base.question, lowercase)
                    if name.casefold() in lowered_vocabulary
                ]
                base.meta["names"] = names
            if names:
                with_actor += 1
            distinct.update(names)
        report["pools"][pool] = {
            "bases": len(bases),
            "bases_with_actor": with_actor,
            "coverage": round(with_actor / len(bases), 4) if bases else 0.0,
            "distinct_actors": len(distinct),
            "top_actors": [name for name, _ in distinct.most_common(10)],
        }
    return report


def load_sum_answerable(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR, *, split: str = "train"
) -> list[BaseQuestion]:
    """SUM's answerable side.  Same row as the unanswerable variant -> same distribution."""
    import pyarrow.parquet as pq

    path = Path(raw_dir) / "sum" / f"{split}.parquet"
    table = pq.read_table(path, columns=["answerable_question", "ground_truth"])
    questions = table.column("answerable_question").to_pylist()
    answers = table.column("ground_truth").to_pylist()
    bases: list[BaseQuestion] = []
    for index, (question, answer) in enumerate(zip(questions, answers)):
        question = (question or "").strip()
        if not question:
            continue
        bases.append(
            BaseQuestion(
                question=question,
                answer=(answer or "").strip(),
                pool=POOL_SUM,
                uid=f"sum:{split}:{index}",
                split=split,
                index=index,
            )
        )
    return bases


def load_umwp_answerable(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR,
) -> list[BaseQuestion]:
    """UMWP's answerable half of the 2,600 pairs (``answerable == "True"``)."""
    path = Path(raw_dir) / "umwp" / "StandardDataset.jsonl"
    bases: list[BaseQuestion] = []
    with path.open(encoding="utf-8") as handle:
        # The physical line is the key: upstream's ``id`` is unique over today's
        # 2,600 answerable rows, but a row that ever ships without one would fall
        # back to the same sentinel and collide.
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("answerable", "")).strip().lower() != "true":
                continue
            question = (row.get("question") or "").strip()
            if not question:
                continue
            values = _answer_values(row.get("answer"))
            if not values or values[0] is None:
                continue
            bases.append(
                BaseQuestion(
                    question=question,
                    answer=_render_number(values[0]),
                    pool=POOL_UMWP,
                    uid=f"umwp:{line_number}",
                    split="train",
                    index=int(row.get("id", -1)),
                )
            )
    return bases


def _answer_values(raw: Any) -> list[Any]:
    """UMWP's ``answer`` as a list.

    The field is a JSON *array* in the parsed file (``[460.0]``) even though the
    upstream repo documents it as a string; older dumps do hand back the repr, so
    both are accepted rather than silently dropping all 2,600 answerable rows.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return raw if isinstance(raw, list) else [raw]


def _render_number(value: Any) -> str:
    """Render a numeric answer the way the answer contract expects it."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def kk_answer(names: Sequence[str], solution: Sequence[bool], knight_knave: dict) -> str:
    """K&K answer words in ``names`` order, using the row's own role words.

    Must stay identical to whatever ``kk_adapter.py`` emits -- the reward maps the
    model's answer through ``role_words`` and compares word by word, so a
    divergence here silently mis-scores the 400 synthesised rows.
    """
    truth_word = knight_knave.get("knight") or "knight"
    lie_word = knight_knave.get("knave") or "knave"
    return " ".join(truth_word if flag else lie_word for flag in solution)


def _literal(value: Any) -> Any:
    """A list/dict that upstream stores either typed or as a Python repr.

    Which one it is depends on the endpoint: the datasets-server parquet gives
    real ``list<string>`` / ``list<bool>`` / struct columns, while a raw JSONL
    dump hands back reprs.  Both are accepted, because silently dropping every row
    of a whole source is the failure mode this is here to prevent.
    """
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_kk_clean(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR, *, min_inhabitants: int = 4
) -> list[BaseQuestion]:
    """K&K clean puzzles with at least ``min_inhabitants`` people (design doc section 4.8)."""
    import pyarrow.parquet as pq

    kk_dir = Path(raw_dir) / "kk"
    bases: list[BaseQuestion] = []
    for path in sorted(kk_dir.glob("clean__train__*ppl.parquet")):
        table = pq.read_table(
            path, columns=["quiz", "names", "knight_knave", "solution", "index"]
        )
        for row in table.to_pylist():
            names = _literal(row["names"])
            solution = _literal(row["solution"])
            knight_knave = _literal(row["knight_knave"])
            if not isinstance(names, list) or not isinstance(solution, list):
                continue
            if not isinstance(knight_knave, dict):
                continue
            if len(names) < min_inhabitants or len(names) != len(solution):
                continue
            question = (row["quiz"] or "").strip()
            if not question:
                continue
            bases.append(
                BaseQuestion(
                    question=question,
                    answer=kk_answer(names, solution, knight_knave),
                    pool=POOL_KK,
                    # The file is part of the key: `index` restarts at 0 in each of
                    # the five per-size files, so it repeats five times over here.
                    uid=f"kk:{path.stem}:{row.get('index')}",
                    split="train",
                    index=int(row.get("index", -1)),
                    role_words=(
                        knight_knave.get("knight") or "knight",
                        knight_knave.get("knave") or "knave",
                    ),
                    meta={"names": list(names)},
                )
            )
    return bases


def load_main_pool(
    stage1_path: str | os.PathLike, *, limit: int | None = None
) -> list[BaseQuestion]:
    """The stage-1 math pool, as the 4th synthesised slice (design doc section 4.9.4).

    Stage-1 rows carry no separate question column: their ``prompt`` *is* the
    problem, and ``reward_model.ground_truth`` is the JSON payload the reward
    reads.  Only rows the stage-1 pipeline marked solvable are eligible.
    """
    import pyarrow.parquet as pq

    bases: list[BaseQuestion] = []
    for row_number, row in enumerate(pq.read_table(stage1_path).to_pylist()):
        prompt = row.get("prompt") or []
        if not prompt:
            continue
        question = (prompt[0].get("content") or "").strip()
        if not question:
            continue
        try:
            payload = json.loads(row["reward_model"]["ground_truth"])
        except (ValueError, KeyError, TypeError):
            continue
        answer = payload.get("answer")
        if not answer:
            continue
        bases.append(
            BaseQuestion(
                question=question,
                answer=str(answer),
                pool=POOL_MAIN,
                uid=f"main:{row_number}",
                split=str((row.get("extra_info") or {}).get("split", "train")),
                index=len(bases),
            )
        )
        if limit is not None and len(bases) >= limit:
            break
    return bases


# ---------------------------------------------------------------------------
# rendering one candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Distractor:
    """A rendered distractor sentence plus the labels it earns on this base."""

    role: str
    number: str
    sentence: str
    role_label: str
    number_label: str
    sentence_label: str
    template: str

    @property
    def cell(self) -> tuple[str, str, str]:
        return (self.role_label, self.number_label, self.sentence_label)


@dataclass(frozen=True)
class Candidate:
    """A distractor bound to the base problem it will be attached to."""

    base: BaseQuestion
    distractor: Distractor
    question: str
    order: float

    @property
    def cell(self) -> tuple[str, str, str]:
        return self.distractor.cell


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def inject(question: str, sentence: str) -> str:
    """Insert ``sentence`` the way GSM-IC does: just before the final question.

    GSM-IC's own ``new_question`` puts the distractor sentence second-to-last, so
    the problem still ends on its question.  When the base does not end in a
    question, the sentence is appended.

    Whitespace is collapsed per sentence.  A SUM/MATH question often spans lines
    ("... degree 3n such that\\nP(0) = P(3) = ..."), and ``re.split`` only breaks
    where a ``.!?`` precedes the gap, so a bare newline would otherwise survive
    inside a part and make the base text no longer a substring of the rendered
    prompt -- which is exactly what :func:`verify_row` checks.
    """
    parts = [
        " ".join(part.split())
        for part in _SENTENCE_SPLIT.split(question.strip())
        if part.strip()
    ]
    sentence = " ".join(sentence.split())
    if not parts:
        return sentence
    if "?" in parts[-1]:
        return " ".join([*parts[:-1], sentence, parts[-1]])
    return " ".join([*parts, sentence])


def choose_distractor(
    template: Template,
    base: BaseQuestion,
    rng: random.Random,
    *,
    names: Sequence[str] | None = None,
    numbers: Sequence[str] | None = None,
    want_role: str | None = None,
    want_number: str | None = None,
    want_sentence: str | None = None,
) -> Distractor | None:
    """Fill one template against one base, or ``None`` if it cannot be made safe.

    ``names`` and ``numbers`` are the pools of *substitutable* roles and values
    (GSM-IC's own 457 roles / 58 numbers by default).  Each axis can be requested
    explicitly -- ``overlapped``/``nonoverlapped``, ``in_range``/``out_range``,
    ``in_topic``/``out_topic`` -- in which case this returns ``None`` rather than
    a differently-labelled distractor.  The label is always *recomputed* from the
    text that was actually built, never taken from the request, so a caller can
    only ever get the cell it asked for or nothing.

    Two requests are structurally unsatisfiable and the sampler above this knows
    it: an ``overlapped`` role may not reuse any content word of the base (that
    would restate a known property and contradict it), so ``overlapped`` implies
    ``out_topic``; and a base with no numbers at all can only produce
    ``out_range``, which is why the K&K pool reports an ``in_range`` shortfall
    instead of faking those rows.
    """
    local_names = list(base_names_of(base))
    requested_role = want_role
    want_role = want_role or rng.choice(("overlapped", "nonoverlapped"))
    if want_role == "overlapped" and not local_names:
        # An explicit request that cannot be met returns nothing rather than a
        # row that would land in a different cell than the caller asked for.
        if requested_role:
            return None
        want_role = "nonoverlapped"
    if want_role == "nonoverlapped" and not (names or []):
        if requested_role or not local_names:
            return None
        want_role = "overlapped"

    lowered_base = base.question.casefold()
    if want_role == "overlapped":
        pool = list(local_names)
        role_label = "overlapped"
    else:
        pool = [n for n in (names or []) if n.casefold() not in lowered_base]
        role_label = "nonoverlapped"
    if not pool:
        return None

    numbers = list(numbers or [])
    base_values = _float_numbers(base.question)
    want_number = want_number or ("in_range" if base_values else "out_range")
    if want_number == "in_range":
        if not base_values:
            # Nothing to be inside of, so this base cannot carry an in_range fill.
            return None
        candidates = [n for n in numbers if _in_range(n, base_values)]
        if not candidates:
            candidates = [_whole_number_in(base_values, rng)]
        number = rng.choice(candidates)
        number_label = "in_range"
    else:
        candidates = [n for n in numbers if not _in_range(n, base_values)]
        if not candidates:
            return None
        number = rng.choice(candidates)
        number_label = "out_range"

    # `pool` is in document order, and the actor of a word problem is named early.
    role = rng.choice(pool[:3] if role_label == "overlapped" else pool)
    base_words = _base_content_words(base.question)
    overlap = template.content_words & base_words
    sentence_label = "in_topic" if template.topic_words & base_words else "out_topic"
    if role_label == "overlapped" and overlap:
        # Restating a property the base already fixes would contradict it.
        return None
    if want_sentence is not None and sentence_label != want_sentence:
        return None

    sentence = template.render(role, number)
    if not sentence or sentence.casefold() in lowered_base:
        return None
    return Distractor(
        role=role,
        number=number,
        sentence=sentence,
        role_label=role_label,
        number_label=number_label,
        sentence_label=sentence_label,
        template=template.sentence_template,
    )


def _whole_number_in(values: Sequence[float], rng: random.Random) -> str:
    """A whole number inside the base's numeric span, for ``in_range`` fills.

    GSM-IC's own number pool rarely lands inside a given MATH problem's range, so
    the guard needs a fallback -- but it must not emit ``6.1382...``, which no
    GSM-IC sentence ever contains.
    """
    low, high = min(values), max(values)
    whole = [value for value in (int(low), int(low) + 1, int(high)) if low <= value <= high]
    if whole:
        return str(rng.choice(sorted(set(whole))))
    return _render_number(low)


def _in_range(number: str, values: Sequence[float]) -> bool:
    if not values:
        return False
    try:
        value = float(number)
    except ValueError:
        return False
    low, high = min(values), max(values)
    return low <= value <= high


def base_names_of(base: BaseQuestion) -> list[str]:
    """Actors available for an *overlapped* fill on this base.

    For K&K the puzzle's universe is closed -- only its listed inhabitants -- and
    for every other pool the list was mined once by :func:`mine_actors`.  The empty
    default is deliberate: an unmined base loses its overlapped cell rather than
    falling back to a capitalised verb (see the actor-mining note above).
    """
    return list(base.meta.get("names") or [])


def _base_content_words(text: str) -> set[str]:
    return {
        token.casefold()
        for token in schema.words(text)
        if schema.is_content_word(token) and not token.isdigit()
    }


class BaseIndex:
    """Bases plus the two lookups an intent-directed sampler needs.

    ``names`` and ``word_to_ids`` are what turn sampling from "draw a random pair
    and hope" into "draw a pair that has the label I want".  Without them the
    ``in_topic`` axis collapses: GSM-IC's templates are about tomatoes and shoe
    sizes, a MATH problem is about windows and panes, and a random pairing almost
    never shares a topic word.
    """

    def __init__(self, bases: Sequence[BaseQuestion]) -> None:
        self.bases = list(bases)
        self.names: dict[int, list[str]] = {}
        self.word_to_ids: dict[str, list[int]] = defaultdict(list)
        for identifier, base in enumerate(self.bases):
            names = base_names_of(base)
            if names:
                self.names[identifier] = names
            for word in _base_content_words(base.question):
                self.word_to_ids[word].append(identifier)
        self.ids = list(range(len(self.bases)))
        self.used: set[int] = set()

    def draw(
        self,
        template: Template,
        role_label: str,
        sentence_label: str,
        rng: random.Random,
    ) -> BaseQuestion | None:
        """An unused base that can carry ``template`` with the requested labels."""
        pool: Sequence[int]
        if sentence_label == "in_topic":
            words = sorted(template.topic_words & set(self.word_to_ids))
            if not words:
                return None
            pool = self.word_to_ids[rng.choice(words)]
        else:
            pool = self.ids
        if not pool:
            return None
        for _ in range(12):
            identifier = pool[rng.randrange(len(pool))]
            if identifier in self.used:
                continue
            if role_label == "overlapped" and identifier not in self.names:
                continue
            return self.bases[identifier]
        return None

    def mark_used(self, base: BaseQuestion) -> None:
        """Bases are compared by identity, so this cannot be fooled by equal text."""
        for identifier, candidate in enumerate(self.bases):
            if candidate is base:
                self.used.add(identifier)
                return


def impossible_cells() -> set[tuple[str, str, str]]:
    """Cells no distractor can ever occupy, whatever the data looks like.

    An ``overlapped`` role reuses a name the base already mentions, so the
    sentence may not reuse any of the base's content words -- that would restate
    a property the base has already fixed and contradict it.  ``in_topic`` is
    *defined* as reusing a content word, so the two cannot both hold.  This is a
    property of the guard, not of the inventory: it holds for every source.
    """
    return {
        ("overlapped", number_label, "in_topic")
        for number_label in AXIS_TARGETS["number_label"]
    }


def plan_cells(
    quota: int,
    targets: dict[str, dict[str, float]] | None = None,
    impossible: Iterable[tuple[str, str, str]] = (),
) -> dict[tuple[str, str, str], int]:
    """Split ``quota`` over the eight label cells so the *marginals* hold exactly.

    Planning the product of the three marginals per cell and hoping is not enough:
    any cell the structural guard forbids would then be silently unbuildable, and
    its share would spill into whichever cell happened to have supply left, moving
    the marginals with it.  Instead the role x sentence block is filled greedily
    over the *allowed* cells only (largest ``min`` of the two residuals first),
    which reproduces the requested marginals exactly whenever they are jointly
    attainable -- 50% overlapped needs 50% out_topic, so it fits inside 55% -- and
    leaves a shortfall to be reported when they are not.  The number axis is then
    split inside each block proportionally, since it couples to nothing.
    """
    targets = targets or AXIS_TARGETS
    blocked = set(impossible)
    role_budget = _largest_remainder(targets["role_label"], quota)
    sentence_budget = _largest_remainder(targets["sentence_label"], quota)
    number_budget = _largest_remainder(targets["number_label"], quota)

    number_labels = list(targets["number_label"])
    remaining_role = dict(role_budget)
    remaining_sentence = dict(sentence_budget)
    blocks: dict[tuple[str, str], int] = {}
    while True:
        best: tuple[int, str, str] | None = None
        for role_label, role_left in remaining_role.items():
            if role_left <= 0:
                continue
            for sentence_label, sentence_left in remaining_sentence.items():
                if sentence_left <= 0:
                    continue
                # Impossibility in this planner is per (role, sentence) -- both of
                # the rules that produce it are independent of the number axis.
                if all((role_label, n, sentence_label) in blocked for n in number_labels):
                    continue
                take = min(role_left, sentence_left)
                if best is None or take > best[0]:
                    best = (take, role_label, sentence_label)
        if best is None or best[0] <= 0:
            break
        take, role_label, sentence_label = best
        key = (role_label, sentence_label)
        blocks[key] = blocks.get(key, 0) + take
        remaining_role[role_label] -= take
        remaining_sentence[sentence_label] -= take

    cells: dict[tuple[str, str, str], int] = {}
    for (role_label, sentence_label), total in blocks.items():
        for number_label, count in _largest_remainder(number_budget, total).items():
            if count > 0:
                cells[(role_label, number_label, sentence_label)] = count
    return cells


def build_candidates(
    pool: str,
    bases: Sequence[BaseQuestion],
    templates: Sequence[Template],
    want: int,
    rng: random.Random,
    *,
    oversample: int = 3,
    role_pool: Sequence[str] = (),
    number_pool: Sequence[str] = (),
    order_start: int = 0,
    targets: dict[str, dict[str, float]] | None = None,
) -> tuple[list[Candidate], dict[tuple[str, str, str], int]]:
    """Render roughly ``want * oversample`` candidates for one pool.

    Returns the candidates and the per-cell plan they were drawn against, so the
    allocator can consume the plan instead of re-deriving a different one.

    Generation is driven by :func:`plan_cells`: each cell is asked for its share
    of the quota, and the sampler draws only pairs whose *recomputed* labels land
    in that cell.  Each base is used at most once, so the synthesised rows cannot
    duplicate a base problem; when a cell's supply runs out the caller sees an
    empty cell and the allocator reports the drift rather than a skewed sample
    being passed off as balanced.
    """
    index = BaseIndex(bases)
    # Pool-specific impossibilities, derived from the data rather than hard-coded
    # per source: a pool whose bases name nobody cannot carry an overlapped fill,
    # and one whose bases carry no numbers cannot carry an in_range fill.
    impossible = set(impossible_cells())
    if not index.names:
        impossible |= {
            (role_label, number_label, sentence_label)
            for role_label in AXIS_TARGETS["role_label"]
            for number_label in AXIS_TARGETS["number_label"]
            for sentence_label in AXIS_TARGETS["sentence_label"]
            if role_label == "overlapped"
        }
    if not any(_float_numbers(base.question) for base in index.bases):
        impossible |= {
            (role_label, number_label, sentence_label)
            for role_label in AXIS_TARGETS["role_label"]
            for number_label in AXIS_TARGETS["number_label"]
            for sentence_label in AXIS_TARGETS["sentence_label"]
            if number_label == "in_range"
        }

    cells = plan_cells(want, targets, impossible)
    order = iter(range(order_start, order_start + 10**9))
    candidates: list[Candidate] = []
    for cell, need in sorted(cells.items()):
        if need <= 0:
            continue
        role_label, number_label, sentence_label = cell
        goal = max(need * oversample, need)
        attempts, built = 0, 0
        limit = goal * 40 + 200
        while built < goal and attempts < limit:
            attempts += 1
            template = templates[rng.randrange(len(templates))]
            base = index.draw(template, role_label, sentence_label, rng)
            if base is None:
                continue
            distractor = choose_distractor(
                template,
                base,
                rng,
                names=role_pool,
                numbers=number_pool,
                want_role=role_label,
                want_number=number_label,
                want_sentence=sentence_label,
            )
            if distractor is None:
                continue
            index.mark_used(base)
            candidates.append(
                Candidate(
                    base=base,
                    distractor=distractor,
                    question=inject(base.question, distractor.sentence),
                    order=float(next(order)),
                )
            )
            built += 1
    return candidates, cells


# ---------------------------------------------------------------------------
# balance-aware allocation
# ---------------------------------------------------------------------------


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion ``total`` over ``weights``, summing to exactly ``total``."""
    mass = sum(weights.values())
    if total <= 0 or mass <= 0:
        return {key: 0 for key in weights}
    exact = {key: total * value / mass for key, value in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(exact, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


@dataclass
class AllocationReport:
    """What the allocator could and could not honour for one pool."""

    pool: str
    requested: int
    selected: int
    shortfall: int
    cells: dict[str, dict[str, int]]
    achieved: dict[str, dict[str, float]]
    targets: dict[str, dict[str, float]]
    balanced: dict[str, bool]
    infeasible_cells: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "requested": self.requested,
            "selected": self.selected,
            "shortfall": self.shortfall,
            "cells": self.cells,
            "achieved": self.achieved,
            "targets": self.targets,
            "balanced": self.balanced,
            "infeasible_cells": self.infeasible_cells,
        }


def allocate(
    candidates: Sequence[Candidate],
    quota: int,
    *,
    cell_targets: dict[tuple[str, str, str], int] | None = None,
    targets: dict[str, dict[str, float]] | None = None,
    tolerance: float = AXIS_TOLERANCE,
    pool: str = "",
) -> tuple[list[Candidate], AllocationReport]:
    """Pick ``quota`` candidates, holding all three axes inside ``tolerance``.

    Three passes.  The first takes each cell's planned count, which is what makes
    the *marginals* come out right: filling cells one-for-one instead would drain
    the ``nonoverlapped`` and ``out_topic`` budgets long before ``overlapped``
    reached its 50%, because the plan deliberately asks for unequal cell counts
    (250 overlapped/out_topic against 25 nonoverlapped/out_topic).  The second
    spends whatever budget the first left over, one row per cell per sweep, so a
    cell that ran short is covered by cells that have slack -- but only within the
    axis budgets.  The third fills the rest ignoring the budgets, because the row
    *count* the design doc asks for outranks the balance when the two conflict --
    and the report says which happened, per axis, instead of quietly returning a
    skewed sample.
    """
    targets = targets or AXIS_TARGETS
    budgets = {axis: _largest_remainder(targets[axis], quota) for axis in AXES}
    buckets: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.cell].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda c: c.order)
    # Supply has to be read *before* the passes drain the buckets: afterwards
    # every bucket is short, and a drained bucket is indistinguishable from one
    # that never had a candidate.
    supply = {
        cell: len(buckets.get(cell, ())) for cell in sorted(_all_cells(targets))
    }

    selected: list[Candidate] = []
    if cell_targets:
        for cell in sorted(cell_targets):
            if len(selected) >= quota:
                break
            bucket = buckets.get(cell)
            if not bucket:
                continue
            for _ in range(min(cell_targets[cell], len(bucket), quota - len(selected))):
                selected.append(bucket.pop())
                for axis, label in zip(AXES, cell):
                    budgets[axis][label] -= 1

    progress = True
    while progress and len(selected) < quota:
        progress = False
        for cell in sorted(buckets):
            if len(selected) >= quota:
                break
            bucket = buckets[cell]
            if not bucket:
                continue
            if any(budgets[axis][label] <= 0 for axis, label in zip(AXES, cell)):
                continue
            selected.append(bucket.pop())
            for axis, label in zip(AXES, cell):
                budgets[axis][label] -= 1
            progress = True

    if len(selected) < quota:
        for cell in sorted(buckets):
            while buckets[cell] and len(selected) < quota:
                selected.append(buckets[cell].pop())

    selected.sort(key=lambda c: c.order)
    counts = {
        axis: Counter(getattr(candidate.distractor, axis) for candidate in selected)
        for axis in AXES
    }
    total = len(selected)
    achieved = {
        axis: {
            label: (counts[axis].get(label, 0) / total if total else 0.0)
            for label in targets[axis]
        }
        for axis in AXES
    }
    balanced = {
        axis: all(
            abs(achieved[axis][label] - fraction) <= tolerance
            for label, fraction in targets[axis].items()
        )
        for axis in AXES
    }
    cells = {"/".join(cell): count for cell, count in supply.items() if count}
    infeasible = ["/".join(cell) for cell, count in supply.items() if not count]
    report = AllocationReport(
        pool=pool,
        requested=quota,
        selected=total,
        shortfall=max(0, quota - total),
        cells=cells,
        achieved=achieved,
        targets={axis: dict(targets[axis]) for axis in AXES},
        balanced=balanced,
        infeasible_cells=infeasible,
    )
    return selected, report


def _all_cells(targets: dict[str, dict[str, float]]) -> set[tuple[str, str, str]]:
    cells = {()}
    for axis in AXES:
        cells = {cell + (label,) for cell in cells for label in targets[axis]}
    return cells  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# row emission
# ---------------------------------------------------------------------------


def make_rows(selected: Sequence[Candidate]) -> list[dict]:
    """Turn selected candidates into parquet rows (template B, gold unchanged)."""
    rows: list[dict] = []
    for candidate in selected:
        base = candidate.base
        distractor = candidate.distractor
        role_words = list(base.role_words)
        ground_truth = schema.build_ground_truth(
            solvable=True,
            answer=base.answer,
            perturbation_type=DISTRACTOR_PERTURBATION,
            role_words=role_words or None,
        )
        extra_info: dict[str, Any] = {
            "split": base.split,
            "index": base.index,
            # Unique per pool: the base is used at most once, and ``uid`` is the
            # pool-unique upstream key (`index` alone collides -- see BaseQuestion).
            "task_id": f"d17:{base.uid}",
            "solvable": True,
            # Mirrors the ground_truth payload.  The reward reads the payload, but
            # D18's monitoring groups by this column, and an empty one silently
            # hides these 2,400 rows from the perturbation breakdown.
            "perturbation_type": DISTRACTOR_PERTURBATION,
            "paired_original_text": base.question,
            "distractor_text": distractor.sentence,
            "distractor_labels": {
                "role_label": distractor.role_label,
                "number_label": distractor.number_label,
                "sentence_label": distractor.sentence_label,
            },
            # The source template is deliberately *not* a row field: extra_info is
            # one Arrow struct shared with every other source, and the D17 promise
            # that has to be auditable per row is the answer invariance and the
            # three labels, not which of the 200-odd templates was drawn.  The
            # report keeps the per-cell template usage instead.
            "perturbation_family": "gsmic_template",
            "role_words": role_words,
        }
        rows.append(
            schema.make_row(
                data_source=POOL_DATA_SOURCE[base.pool],
                question=candidate.question,
                ground_truth=ground_truth,
                template=schema.TEMPLATE_B,
                branch=POOL_BRANCH[base.pool],
                extra_info=extra_info,
                role_words=role_words or None,
            )
        )
    return rows


def verify_row(base: BaseQuestion, row: dict) -> list[str]:
    """Audit one synthesised row against the base it came from.

    The D17 promise is narrow and checkable: the base problem is untouched, the
    gold answer is byte-identical, and the added sentence is a red herring the
    answer does not depend on.  This checks the first two exactly and the third
    structurally (the sentence is present, and deleting it from the prompt gives
    the base problem back).
    """
    problems: list[str] = []
    info = row["extra_info"]
    distractor = info.get("distractor_text") or ""
    # Whitespace is normalised before comparing: `inject` re-joins sentences on a
    # single space, so a base question with a double space ("5'6\".  He grows")
    # would otherwise look like it had been rewritten.
    haystack = " ".join(row["prompt"][0]["content"].split())
    payload = json.loads(row["reward_model"]["ground_truth"])
    if payload.get("answer") != base.answer:
        problems.append("gold answer changed by synthesis")
    if not payload.get("solvable"):
        problems.append("synthesised row is not marked solvable")
    if payload.get("perturbation_type") != DISTRACTOR_PERTURBATION:
        problems.append(f"unexpected perturbation_type {payload.get('perturbation_type')!r}")
    if info.get("paired_original_text") != base.question:
        problems.append("paired_original_text does not carry the base question")
    if not distractor:
        problems.append("row has no distractor_text")
    elif " ".join(distractor.split()) not in haystack:
        problems.append("distractor sentence is not in the rendered prompt")
    else:
        # The base is *not* a contiguous span of the prompt -- `inject` puts the
        # sentence second-to-last, so it lands mid-problem by design.  The
        # invariant is that deleting the added sentence restores the base, which
        # also catches a distractor that clipped a neighbouring sentence on its way
        # in.  The comparison is a containment, not an equality, because K&K role
        # rows append the answer-format instruction after the puzzle.
        residual = " ".join(haystack.replace(" ".join(distractor.split()), " ").split())
        if " ".join(base.question.split()) not in residual:
            problems.append("base question text is not recoverable once the distractor is removed")
    labels = info.get("distractor_labels") or {}
    for axis in AXES:
        if labels.get(axis) not in AXIS_TARGETS[axis]:
            problems.append(f"{axis} is {labels.get(axis)!r}")
    return problems


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def load_pools(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR,
    stage1_path: str | os.PathLike | None = None,
) -> tuple[dict[str, list[BaseQuestion]], dict[str, str]]:
    """Load every base pool, recording why any pool came back empty."""
    pools: dict[str, list[BaseQuestion]] = {}
    notes: dict[str, str] = {}
    for pool, loader in (
        (POOL_SUM, load_sum_answerable),
        (POOL_UMWP, load_umwp_answerable),
        (POOL_KK, load_kk_clean),
    ):
        try:
            pools[pool] = loader(raw_dir)
        except FileNotFoundError as exc:
            pools[pool] = []
            notes[pool] = f"raw file missing: {exc}"
    if stage1_path is None:
        pools[POOL_MAIN] = []
        notes[POOL_MAIN] = "no --stage1-path given"
    else:
        try:
            pools[POOL_MAIN] = load_main_pool(stage1_path)
        except FileNotFoundError as exc:
            pools[POOL_MAIN] = []
            notes[POOL_MAIN] = f"stage-1 parquet missing: {exc}"
    return pools, notes


_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")


def allocate_role_budget(
    supply: Mapping[str, int],
    buildable: Mapping[str, int],
    target: float,
) -> dict[str, int]:
    """Split a global overlapped-role budget across pools, capped by what each can build.

    The role axis cannot be planned per pool at the design's 50/50, because the
    pools differ in *supply*: only a base that names a person can carry an
    overlapped fill at all, and the MATH pool names someone on about one base in
    ten.  So the budget is set globally -- ``target`` of everything buildable, or
    less if supply cannot cover it -- and handed out *in proportion to each pool's
    room*.  While the budget stays under the total room every pool therefore lands
    on the same overlapped share; once supply caps it, a pool contributes what it
    has and the pools that can still reach the target do.  (An earlier revision
    gave each row to the pool with the most room left, which handed the whole
    budget to the two large pools and left the smallest at 0.0 overlapped.)

    ``buildable[pool]`` is what the pool can actually emit -- its quota, or its
    number of bases if that is smaller -- and ``room`` is capped by both supply
    and the ``sentence_label`` target, since an overlapped fill is necessarily
    out-of-topic (see :func:`impossible_cells`): a pool can never make more
    overlapped rows than it has out_topic budget.  Both caps have to be in terms
    of buildable rows.  Taking the fraction of the *requested* quota instead asks
    for 50% of rows that will never exist, which pins every pool to the out_topic
    cap and lands the axis at 0.55 rather than 0.50 whenever a pool has no source.

    Returns rows per pool; the caller turns them into per-pool fractions.
    """
    out_topic = AXIS_TARGETS["sentence_label"]["out_topic"]
    room = {
        pool: max(0, min(supply.get(pool, 0), int(buildable.get(pool, 0) * out_topic)))
        for pool in supply
    }
    budget = min(round(target * sum(buildable.values())), sum(room.values()))
    return _largest_remainder({pool: float(value) for pool, value in room.items()}, budget)


def synthesise(
    pools: dict[str, list[BaseQuestion]],
    *,
    templates: Sequence[Template],
    quota: dict[str, int] | None = None,
    seed: int = 42,
    mine: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Run the whole D17 stage: candidates, allocation, rows, and a report."""
    rng = random.Random(seed)
    quota = dict(quota or DEFAULT_POOL_QUOTA)
    mining_report = mine_actors(pools) if mine else None
    role_pool = sorted({t.role for t in templates if t.role})
    # The substitutes have to be numbers: the mining records a placeholder it
    # could not fill as the literal "n/a", and interpolating that produced
    # "Proposed fed n/a monkeys." in an earlier revision.
    number_pool = sorted(
        {t.number for t in templates if t.number and _NUMBER_RE.match(t.number)}
    )
    non_numeric_fills = sorted(
        {t.number for t in templates if t.number and not _NUMBER_RE.match(t.number)}
    )

    # The role axis is planned against what the pools can actually supply; every
    # other axis is planned against the design targets per pool.
    supply = {
        pool: sum(1 for base in (pools.get(pool) or []) if base_names_of(base))
        for pool in POOLS
    }
    # What each pool can actually emit.  The role budget is a fraction of this,
    # not of the requested quota: the two differ whenever a pool is short of
    # bases, and the fractions below have to divide by the same denominator.
    buildable = {
        pool: min(int(quota.get(pool, 0)), len(pools.get(pool) or [])) for pool in POOLS
    }
    role_budget = allocate_role_budget(
        supply, buildable, AXIS_TARGETS["role_label"]["overlapped"]
    )
    pool_targets: dict[str, dict[str, dict[str, float]]] = {}
    for pool in POOLS:
        want = int(quota.get(pool, 0))
        targets = {axis: dict(AXIS_TARGETS[axis]) for axis in AXES}
        if buildable.get(pool, 0) > 0:
            fraction = min(1.0, role_budget.get(pool, 0) / buildable[pool])
            targets["role_label"] = {
                "overlapped": fraction,
                "nonoverlapped": 1.0 - fraction,
            }
        pool_targets[pool] = targets

    rows: list[dict] = []
    pool_reports: dict[str, Any] = {}
    order_cursor = 0
    for pool in POOLS:
        want = int(quota.get(pool, 0))
        bases = list(pools.get(pool) or [])
        if want <= 0:
            continue
        if not bases:
            pool_reports[pool] = AllocationReport(
                pool=pool,
                requested=want,
                selected=0,
                shortfall=want,
                cells={},
                achieved={axis: {label: 0.0 for label in AXIS_TARGETS[axis]} for axis in AXES},
                targets={axis: dict(AXIS_TARGETS[axis]) for axis in AXES},
                balanced={axis: False for axis in AXES},
                infeasible_cells=[],
            ).as_dict()
            continue
        # (the empty-pool report above uses the design targets: nothing was built,
        # so there is no supply-capped target to report against)
        candidates, cell_plan = build_candidates(
            pool,
            bases,
            templates,
            want,
            rng,
            role_pool=role_pool,
            number_pool=number_pool,
            order_start=order_cursor,
            targets=pool_targets[pool],
        )
        selected, report = allocate(
            candidates, want, cell_targets=cell_plan, targets=pool_targets[pool], pool=pool
        )
        rows.extend(make_rows(selected))
        pool_reports[pool] = report.as_dict()
        order_cursor += len(candidates) + 1

    total = len(rows)
    overall = {
        axis: {
            label: (
                sum(1 for row in rows if row["extra_info"]["distractor_labels"][axis] == label)
                / total
                if total
                else 0.0
            )
            for label in AXIS_TARGETS[axis]
        }
        for axis in AXES
    }
    report: dict[str, Any] = {
        "seed": seed,
        "quota": quota,
        "requested_total": sum(quota.get(pool, 0) for pool in POOLS),
        "built_total": total,
        "pools": pool_reports,
        "achieved_overall": overall,
        "balanced_overall": {
            axis: all(
                abs(overall[axis][label] - fraction) <= AXIS_TOLERANCE
                for label, fraction in AXIS_TARGETS[axis].items()
            )
            for axis in AXES
        },
        "axes": AXES,
        "tolerance": AXIS_TOLERANCE,
        "design_targets": {axis: dict(AXIS_TARGETS[axis]) for axis in AXES},
        "effective_targets": {
            pool: {
                "targets": pool_targets[pool],
                "buildable": buildable[pool],
                "overlapped_supply": supply[pool],
                "overlapped_budget": role_budget.get(pool, 0),
            }
            for pool in POOLS
        },
        "actor_mining": mining_report,
        "number_pool_size": len(number_pool),
        "non_numeric_fills_dropped": non_numeric_fills,
    }
    # The role axis is the one axis whose design target is a *claim* the data can
    # refuse: state the refusal in the artefact, in one line, rather than leaving
    # it to be inferred from a marginal that quietly came out at 0.44.
    design_role = AXIS_TARGETS["role_label"]["overlapped"]
    buildable_total = sum(buildable.values())
    built_supply = sum(supply.get(pool, 0) for pool in POOLS if quota.get(pool, 0) > 0)
    effective_role = sum(role_budget.values())
    requested_total = sum(quota.values())
    report["role_supply_note"] = (
        f"overlapped role supply {built_supply} of {buildable_total} buildable rows "
        f"({requested_total} requested); "
        f"budget {effective_role} against a design target of "
        f"{round(design_role * buildable_total)} "
        f"({effective_role / buildable_total:.3f} vs {design_role:.2f})"
        if buildable_total
        else "nothing buildable"
    )
    return rows, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--stage1-path", default=None, help="stage-1 math parquet")
    parser.add_argument("--out", default=DEFAULT_OUT, help="write rows here as parquet")
    parser.add_argument(
        "--report", default=None, help="JSON report path (default: alongside --out)"
    )
    parser.add_argument("--sum-quota", type=int, default=DEFAULT_POOL_QUOTA[POOL_SUM])
    parser.add_argument("--umwp-quota", type=int, default=DEFAULT_POOL_QUOTA[POOL_UMWP])
    parser.add_argument("--kk-quota", type=int, default=DEFAULT_POOL_QUOTA[POOL_KK])
    parser.add_argument("--main-quota", type=int, default=DEFAULT_POOL_QUOTA[POOL_MAIN])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0

    templates, template_report = load_safe_templates(args.raw_dir)
    if not templates:
        print("no safe templates -- cannot synthesise", file=sys.stderr)
        return 1
    print(
        f"templates: {template_report['safe']} safe / {template_report['distinct_total']} "
        f"distinct  ({template_report['rejected_by_reason']})"
    )

    pools, notes = load_pools(args.raw_dir, args.stage1_path)
    for pool in POOLS:
        note = notes.get(pool, "")
        print(f"pool {pool:<5} bases={len(pools.get(pool) or []):>7}  {note}")

    quota = {
        POOL_SUM: args.sum_quota,
        POOL_UMWP: args.umwp_quota,
        POOL_KK: args.kk_quota,
        POOL_MAIN: args.main_quota,
    }
    rows, report = synthesise(pools, templates=templates, quota=quota, seed=args.seed)
    report["templates"] = template_report
    report["pool_notes"] = notes

    problems = 0
    for pool in POOLS:
        pool_report = report["pools"].get(pool)
        if not pool_report:
            continue
        print(
            f"built {pool:<5} {pool_report['selected']:>6}/{pool_report['requested']:<6} "
            f"shortfall={pool_report['shortfall']:<5} "
            f"balanced={ {k: v for k, v in pool_report['balanced'].items()} }"
        )
        if pool_report["infeasible_cells"]:
            print(f"    unreachable cells: {pool_report['infeasible_cells']}")
        problems += pool_report["shortfall"]

    print(f"\ntotal {report['built_total']}/{report['requested_total']}")
    for axis in AXES:
        achieved = report["achieved_overall"][axis]
        flag = "OK " if report["balanced_overall"][axis] else "OFF"
        pairs = " ".join(f"{label}={value:.3f}" for label, value in achieved.items())
        print(f"  {flag} {axis:<15} {pairs}")

    if args.out and rows:
        schema.write_rows_parquet(rows, args.out)
        print(f"\nrows -> {args.out}")
    # Always written, and named after ``--out``: ``--out`` empty is the one way
    # to build without writing, and then there is nothing to report beside.
    if args.report or args.out:
        report_path = args.report or report_path_for(args.out)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"report -> {report_path}")
    return 0


def _selftest() -> None:
    """Exercise the safety rules and the allocator without touching the network."""
    for unsafe, reason in (
        ("{role} is {number} inches taller than Steve.", "relational"),
        ("{role} has {number} more potatoes than Oli.", "relational"),
        ("The shoe size of {role} is {number}.", None),
        ("{role} bought {number} newspapers.", None),
        # The placeholder count is checked before the question form, because a
        # sentence that carries only one axis cannot be labelled on the other at
        # all -- so the role-only question is reported as such, not as a question.
        ("How many apples does {role} have?", "single_placeholder"),
        ("How many apples does {role} have out of {number}?", "question_form"),
        ("The sun rises in the east.", "no_placeholder"),
        ("{role} and Steve have {number} apples in total.", "additive"),
        ("{role} is {number} years old.", None),
        ("The salary of {role} is ${number} per month.", None),
        # Both of the two remaining rules need a two-placeholder template to be
        # reachable at all: the placeholder count is checked first, so the
        # one-placeholder spellings of them report ``single_placeholder``.
        ("{role} is a knight who guards {number} doors.", "truth_functional"),
        ("{role} is {number} inches away from Oliver.", "hardcoded_name"),
    ):
        got = unsafe_reason(unsafe)
        assert got == reason, f"{unsafe!r}: expected {reason!r}, got {got!r}"
    assert hardcoded_names("{role} is {number} inches taller than Steve.") == ["Steve"]
    assert hardcoded_names("The height of {role} is {number} feet.") == []

    base = BaseQuestion(
        question="Jewel bought 10 magazines to be sold at $3.50 each. How much will she gain?",
        answer="5",
        pool=POOL_SUM,
    )
    template = Template(
        sentence_template="The height of {role} is {number} feet.",
        role="Emma",
        number="8",
        role_label="nonoverlapped",
        number_label="in_range",
        sentence_label="in_topic",
        count=1,
    )
    distractor = choose_distractor(
        template, base, random.Random(0), names=["Emma"], numbers=["8", "1000"]
    )
    assert distractor is not None
    assert distractor.sentence == "The height of Emma is 8 feet."
    injected = inject(base.question, distractor.sentence)
    assert injected.endswith("How much will she gain?")
    assert distractor.sentence in injected

    # An overlapped role restating a known property must be refused: the base
    # already fixes Jewel's magazines, so a second count would contradict it.
    magazines = Template(
        sentence_template="{role} bought {number} newspapers.",
        role="Emma",
        number="8",
        role_label="nonoverlapped",
        number_label="in_range",
        sentence_label="in_topic",
        count=1,
    )
    overlap_base = BaseQuestion(
        question="Jewel bought 10 newspapers. How many did she buy in total?",
        answer="10",
        pool=POOL_SUM,
    )
    refused = 0
    for seed in range(40):
        got = choose_distractor(magazines, overlap_base, random.Random(seed), names=["Jewel"])
        if got is None or got.role_label != "overlapped":
            refused += 1
    assert refused > 0, "overlapped property restatement was never refused"

    candidates = [
        Candidate(
            base=base,
            distractor=Distractor(
                role="Emma",
                number=str(index),
                sentence="sent",
                role_label=role_label,
                number_label=number_label,
                sentence_label=sentence_label,
                template="t",
            ),
            question="q",
            order=float(index),
        )
        for index, (role_label, number_label, sentence_label) in enumerate(
            (r, n, s)
            for r in ("overlapped", "nonoverlapped")
            for n in ("in_range", "out_range")
            for s in ("in_topic", "out_topic")
            for _ in range(50)
        )
    ]
    selected, report = allocate(candidates, 320, pool="selftest")
    assert report.selected == 320, report.selected
    assert report.balanced["number_label"], report.achieved
    print(f"selftest OK (allocator achieved {report.achieved})")


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
