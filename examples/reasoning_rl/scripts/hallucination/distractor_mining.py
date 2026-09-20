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
"""Same-question, equal-length option mining (design doc D13/D15, sections 4.3 and 5.1).

The judgment branches ask the model to pick *which option makes the premise false*
(FalseQA-fake's replacement pair, D21) or which condition is missing (TreeCut
negatives, D26).  That option set is only a real test if the answer cannot be
found by surface heuristics, and the recon audits measured two heuristics that win
outright:

* **Cross-question text.**  When distractors are borrowed from a different
  problem, "the option that does not occur in the question" scores 88.5-100%
  (MiP report section 4.2 measured 626/626 = 100.0% on MiP).  So every distractor
  must be a span of *this* question.
* **Length.**  The longest option was the gold in 34.3% of MiP rows -- near
  chance with sentence-level options, but only because those options were of
  wildly different lengths.  Keeping every option the same token length removes
  the cue instead of hoping it stays weak.

Three rules follow, all enforced here:

1. every span comes from the same question text;
2. every span has the same number of word tokens as the gold span;
3. every span carries at least one content word (no ``"the of a"`` filler).

A span may not overlap the gold span's character range, so the distractors are
genuinely different text rather than a shifted window over the gold.  When fewer
than ``k - 1`` candidates survive, :func:`mine_option_spans` returns ``None`` and
the adapter **drops the row** -- there is deliberately no fallback to
cross-question text, because that fallback is the leak being defended against.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from schema import STOPWORDS, is_content_word, words

# Token spans are matched on this pattern so a span's token count is unambiguous.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


@dataclass(frozen=True)
class Span:
    """A candidate span of a question, with its character offsets.

    ``char_len`` is what the length-closeness tie-break ranks on; the token count
    is the hard filter and is not stored because it equals the gold's by
    construction.
    """

    text: str
    start: int
    end: int
    n_tokens: int

    @property
    def char_len(self) -> int:
        return self.end - self.start


def token_spans(text: str, n_tokens: int) -> list[Span]:
    """Every window of ``n_tokens`` consecutive word tokens in ``text``.

    Offsets are character positions into ``text``, so the caller can test overlap
    against the gold span.  The window text is sliced from the original string,
    preserving internal spacing and punctuation rather than re-joining tokens.
    """
    if n_tokens <= 0:
        return []
    matches = list(_TOKEN_RE.finditer(text))
    spans: list[Span] = []
    for i in range(len(matches) - n_tokens + 1):
        start = matches[i].start()
        end = matches[i + n_tokens - 1].end()
        spans.append(Span(text=text[start:end], start=start, end=end, n_tokens=n_tokens))
    return spans


def find_span(text: str, needle: str) -> Span | None:
    """Locate ``needle`` in ``text`` and return it as a :class:`Span`.

    Uses the first occurrence.  Returns ``None`` when the needle is absent, which
    is the signal that the caller's assumed gold text and the rendered question
    have drifted apart (a normalization bug, not a mining failure).
    """
    start = text.find(needle)
    if start < 0:
        return None
    return Span(
        text=needle,
        start=start,
        end=start + len(needle),
        n_tokens=len(_TOKEN_RE.findall(needle)),
    )


def mine_option_spans(
    question: str,
    gold_text: str,
    k: int = 3,
    rng: random.Random | None = None,
    max_char_ratio: float = 2.0,
) -> tuple[list[str], str] | None:
    """Mine ``k - 1`` distractors for ``gold_text`` from ``question``.

    Args:
        question: the exact question text that will be rendered into the prompt.
        gold_text: the correct option, a substring of ``question``.
        k: total option count, gold included (D15 fixes the default at 3).
        rng: used to sample among equally good candidates; a fixed seed makes the
            build reproducible.  ``None`` means the process-wide RNG.
        max_char_ratio: reject candidates whose character length differs from the
            gold's by more than this factor.  Token count is the hard filter, but
            ``"a"`` and ``"mississippi"`` are both one token -- an option block of
            wildly unequal *character* lengths leaks the answer to a length
            heuristic even though every option has one token.

    Returns:
        ``(options, correct_option_id)`` ready for
        :func:`schema.build_options`'s caller, or ``None`` when the question does
        not contain ``k - 1`` admissible distractors.
    """
    rng = rng or random
    gold = find_span(question, gold_text)
    if gold is None or gold.n_tokens == 0:
        return None

    gold_key = gold_text.strip().casefold()
    gold_char_len = max(gold.char_len, 1)
    candidates: list[Span] = []
    seen: set[str] = {gold_key}
    for span in token_spans(question, gold.n_tokens):
        key = span.text.strip().casefold()
        if key in seen:
            continue
        if span.start < gold.end and gold.start < span.end:
            continue  # overlaps the gold span
        if not any(is_content_word(tok) for tok in _TOKEN_RE.findall(span.text)):
            continue
        if not 1 / max_char_ratio <= span.char_len / gold_char_len <= max_char_ratio:
            continue
        seen.add(key)
        candidates.append(span)

    if len(candidates) < k - 1:
        return None

    # Prefer the candidates whose character length is closest to the gold's: with
    # the ratio filter this only picks among near-equal lengths, but it keeps the
    # block tight when the question offers many same-token-count windows.
    candidates.sort(key=lambda span: (abs(span.char_len - gold_char_len), span.start))
    best_delta = abs(candidates[0].char_len - gold_char_len)
    best_tier = [s for s in candidates if abs(s.char_len - gold_char_len) == best_delta]
    pool = best_tier if len(best_tier) >= k - 1 else candidates
    chosen = rng.sample(pool, k - 1)

    option_texts = [span.text.strip() for span in chosen] + [gold_text.strip()]
    rng.shuffle(option_texts)
    ids = "ABCDEFGH"[: len(option_texts)]
    correct = ids[option_texts.index(gold_text.strip())]
    return option_texts, correct


def spans_are_same_question(question: str, spans: list[str]) -> bool:
    """Whether every span occurs verbatim in ``question`` (the L3 guard).

    Audit helper, used by every adapter that renders an option block: the
    **placeholder** blocks of the solvable rows (UMWP-answerable D24 /
    FalseQA-answerable D27 / TreeCut positives D26) and the four-tier option
    blocks of the diagnosis sources (FalseQA-fake D21 / TreeCut negatives D26).
    Keeping every option a span of this question is the same-question rule above;
    a cross-question option is guessable by its absence from the question text.
    """
    return all(span in question for span in spans)


def _selftest() -> None:
    """Tiny smoke check, also runnable as ``python distractor_mining.py``."""
    question = "Bryan has 9 books and 46 magazines in each of his 10 bookshelves. How many magazines does he have?"
    gold = "10 bookshelves"
    result = mine_option_spans(question, gold, k=3, rng=random.Random(0))
    assert result is not None, "expected a mineable option block"
    options, correct = result
    assert len(options) == 3 and correct in "ABC"
    assert options["ABC".index(correct)] == gold
    assert spans_are_same_question(question, options), options
    assert len({len(o.split()) for o in options}) == 1, options
    print(f"gold={gold!r} options={options} correct={correct}")

    # A one-token gold in a short question cannot yield 2 admissible distractors,
    # and the function must say so rather than inventing cross-question text.
    assert mine_option_spans("What is 2?", "2", k=3) is None
    print("no-candidate case returns None: ok")


if __name__ == "__main__":
    _selftest()
