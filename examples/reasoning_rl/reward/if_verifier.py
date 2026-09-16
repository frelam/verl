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
"""Verifiable-constraint checker for the instruction-following domain.

Implements the Open-Instruct / IFEval / IFBench constraint set that
``nvidia/Nemotron-RL-instruction_following`` is graded with, and that NVIDIA's
NeMo-Gym environment (``resources_servers/instruction_following/app.py``) runs
through ``verifiable_instructions.instructions_registry.INSTRUCTION_DICT``
(https://github.com/abukharin-nv/verifiable-instructions, Apache-2.0, Google
Research Authors — the same published IFEval checker set).

Each constraint is ``{"id": "<category>:<type>", "kwargs": {...}}`` and the
reward follows the environment's default ``grading_mode="binary"`` convention:
1.0 iff EVERY constraint in ``instruction_id_list`` passes, else 0.0.

Coverage
--------
``SUPPORTED_INSTRUCTION_IDS`` lists every id this module can evaluate; it spans
the whole published registry, including the "new" IFBench constraints the
dataset leans on heavily (``paragraphs:*``, ``first_word:*``, ``last_word:*``,
``count:*``, ``copy:*``, ``letters:*``, ``punctuation:punctuation_dot`` …).  An
id outside that set is scored as failed, so an unimplemented id silently pins a
row's reward to 0 forever — ``unsupported_instruction_ids()`` exists so the
data-level self-check can flag exactly that.

Deviations from the reference implementation (all dependency-free, and all
documented rather than accidental):

* ``language:response_language`` / ``change_case:english_*`` use ``langdetect``
  when it is importable — exactly the engine the reference checker uses, and the
  only way to tell the dataset's 30 target languages apart (``pip install
  langdetect``; the package is pure Python, no model data).  Without it they
  fall back to an ASCII-ratio heuristic, which cannot recognise a correct
  answer in a Latin-script language such as Spanish or Vietnamese.  A text with
  no detectable language counts as followed, mirroring the reference's
  ``LangDetectException`` path.
* ``nltk.word_tokenize`` is replaced by ``_word_tokenize`` (a ``\\w+`` /
  single-punctuation tokenizer).  ``split_into_sentences`` is the reference
  regex implementation, ported verbatim.
* Single-keyword kwargs are unwrapped when the dataset stores them as a
  one-element list (e.g. ``count:count_increment_word`` ships
  ``{"keyword1": ["help"]}``, which raises inside the reference
  ``build_description`` and therefore scores 0 for *every* such row).  This can
  only turn a permanently-zero row into a scoreable one.

The checker is self-contained (the only optional third-party import is
``langdetect``, see above) and tolerant of malformed constraints: unknown ids /
bad kwargs return False for that constraint rather than raising, so one bad
sample never kills the reward pass.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_RELATIONS = {
    "less than": lambda a, b: a < b,
    "at least": lambda a, b: a >= b,
    "at most": lambda a, b: a <= b,
    "more than": lambda a, b: a > b,
    "exactly": lambda a, b: a == b,
    "equal to": lambda a, b: a == b,
}

# Reference comparison relations (verifiable_instructions.instructions).
_COMPARISON_RELATION = ("less than", "at least")

# Reference defaults (verifiable_instructions.instructions constants).
_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
_POSTSCRIPT_MARKERS = ("P.S.", "P.P.S")

# Sentence splitting: verbatim port of
# ``verifiable_instructions.instructions_util.split_into_sentences``.
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (port of instructions_util.split_into_sentences)."""
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def _word_tokenize(text: str) -> list[str]:
    """Stand-in for ``nltk.word_tokenize``: words plus single punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text)


def _count_words(text: str) -> int:
    """Reference ``count_words`` = ``RegexpTokenizer(r"\\w+")``."""
    return len(re.findall(r"\w+", text))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _cmp(count: float, relation: str | None, target) -> bool:
    if target is None:
        return False
    fn = _RELATIONS.get((relation or "").strip().lower(), _RELATIONS["at least"])
    return fn(count, target)


def _one(value):
    """Unwrap a one-element list kwarg (several dataset rows store a list)."""
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _markdown_paragraphs(text: str) -> list[str]:
    """Split on the ``***`` markdown divider (reference ParagraphChecker)."""
    return re.split(r"\s?\*\*\*\s?", text)


def _count_markdown_paragraphs(text: str) -> int | None:
    """Number of non-empty ``***``-separated paragraphs, or None on an empty middle."""
    paragraphs = _markdown_paragraphs(text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return None
    return count


def _looks_english(text: str) -> bool:
    """ASCII-ratio fallback for ``langdetect.detect(text) == "en"``.

    Text with no alphabetic characters has no detectable language, which the
    reference checker counts as *followed* (it returns True on
    ``LangDetectException``) — mirror that rather than failing the constraint.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    return sum(1 for c in alpha if ord(c) < 128) / len(alpha) > 0.9


_detector = None
_detector_checked = False


def _detect_language(text: str) -> str | None:
    """``langdetect.detect(text)`` when the optional package is installed."""
    global _detector, _detector_checked
    if not _detector_checked:
        _detector_checked = True
        try:
            from langdetect import detect

            _detector = detect
        except ImportError:
            _detector = None
    if _detector is None:
        return None
    try:
        return _detector(text)
    except Exception:
        return None


def _is_english(text: str) -> bool:
    """``langdetect.detect(text) == "en"`` with the reference's failure policy."""
    detected = _detect_language(text)
    if detected is not None:
        return detected == "en"
    if not any(c.isalpha() for c in text):
        return True  # reference: undetectable text counts as followed
    return _looks_english(text)


def _strip_code_fence(text: str) -> str:
    return (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


# ---------------------------------------------------------------------------
# keywords:*
# ---------------------------------------------------------------------------


def _check_keywords_existence(text: str, kw: dict) -> bool:
    keywords = _as_list(_first_present(kw, "keywords", "keyword"))
    if not keywords:
        return False
    return all(re.search(str(k), text, flags=re.IGNORECASE) for k in keywords)


def _check_keywords_frequency(text: str, kw: dict) -> bool:
    keyword = _one(_first_present(kw, "keyword", "word"))
    if not keyword:
        return False
    n = len(re.findall(str(keyword), text, flags=re.IGNORECASE))
    return _cmp(n, _first_present(kw, "relation"), _first_present(kw, "frequency", "num_occurrences"))


def _check_keywords_forbidden_words(text: str, kw: dict) -> bool:
    words = _as_list(_first_present(kw, "forbidden_words", "keywords", "words"))
    return all(not re.search(r"\b" + str(w) + r"\b", text, flags=re.IGNORECASE) for w in words)


def _check_keywords_letter_frequency(text: str, kw: dict) -> bool:
    """``keywords:letter_frequency`` and ``letters:letter_counting2`` (same checker)."""
    letter = _one(_first_present(kw, "letter"))
    if not letter or len(str(letter)) != 1:
        return False
    letter = str(letter).lower()
    n = text.lower().count(letter)
    return _cmp(n, _first_present(kw, "let_relation", "relation"), _first_present(kw, "let_frequency", "frequency"))


def _check_keywords_word_once(text: str, kw: dict) -> bool:
    keyword = _one(_first_present(kw, "keyword"))
    if not keyword:
        return False
    return len(re.findall(str(keyword), text, flags=re.IGNORECASE)) == 1


def _check_keywords_word_count_different_numbers(text: str, kw: dict) -> bool:
    keyword = _one(_first_present(kw, "keyword"))
    if not keyword:
        return False
    n = len(re.findall(str(keyword), text, flags=re.IGNORECASE))
    return _cmp(n, _first_present(kw, "relation"), _first_present(kw, "frequency"))


def _check_keywords_palindrome(text: str, kw: dict) -> bool:
    return any(word == word[::-1] for word in text.split())


def _check_keywords_keyword_specific_position(text: str, kw: dict) -> bool:
    keyword = _one(_first_present(kw, "keyword"))
    n = _one(_first_present(kw, "n"))
    m = _one(_first_present(kw, "m"))
    if not keyword or not n or not m:
        return False
    sentences = _split_into_sentences(text)
    if len(sentences) < n:
        return False
    words = _word_tokenize(sentences[n - 1])
    if len(words) < m:
        return False
    return words[m - 1] == keyword


def _check_keywords_no_adjacent_consecutive(text: str, kw: dict) -> bool:
    words = text.split()
    for i in range(len(words) - 1):
        first_letter = words[i][0].lower()
        second_letter = words[i + 1][0].lower()
        if len(first_letter) != 1 or len(second_letter) != 1:
            return False
        if ord(second_letter) - ord(first_letter) == 1:
            return False
    return True


def _check_keywords_start_end(text: str, kw: dict) -> bool:
    words = _word_tokenize(text)
    if len(words) < 2:
        return False
    return words[0].lower() == words[-1].lower()


def _check_keywords_exclude_word_harder(text: str, kw: dict) -> bool:
    keyword = _one(_first_present(kw, "keyword"))
    if not keyword:
        return False
    return f" {keyword} " not in text


# ---------------------------------------------------------------------------
# letters:*
# ---------------------------------------------------------------------------


def _check_letters_letter_counting(text: str, kw: dict) -> bool:
    n = len(re.findall(r"[a-zA-Z]", text))
    return _cmp(n, _first_present(kw, "relation"), _one(_first_present(kw, "N", "num_letters")))


# ---------------------------------------------------------------------------
# language:*
# ---------------------------------------------------------------------------


def _check_language_response_language(text: str, kw: dict) -> bool:
    lang = str(_first_present(kw, "language", "lang") or "").lower()
    if not lang:
        return False
    if lang in ("en", "english"):
        return _is_english(text)
    detected = _detect_language(text)
    if detected is not None:
        return detected == lang
    if lang in ("zh", "chinese", "mandarin"):
        return any("一" <= c <= "鿿" for c in text)
    # Fallback only (langdetect absent): the reference asks langdetect for one
    # of ~30 target languages, so without it a Latin-script target (es, vi, …)
    # cannot be recognised; treat mostly-non-ASCII text as the target language
    # and undetectable text as followed, mirroring the reference.
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    return sum(1 for c in alpha if ord(c) >= 128) / len(alpha) > 0.5


# ---------------------------------------------------------------------------
# length_constraints:*
# ---------------------------------------------------------------------------


def _check_length_number_words(text: str, kw: dict) -> bool:
    return _cmp(_count_words(text), _first_present(kw, "relation"), _first_present(kw, "num_words", "num"))


def _check_length_number_sentences(text: str, kw: dict) -> bool:
    n = len(_split_into_sentences(text))
    return _cmp(n, _first_present(kw, "relation"), _first_present(kw, "num_sentences", "num"))


def _check_length_number_paragraphs(text: str, kw: dict) -> bool:
    target = _first_present(kw, "num_paragraphs", "num")
    if target is None:
        return False
    return _count_markdown_paragraphs(text) == target


def _check_length_nth_paragraph_first_word(text: str, kw: dict) -> bool:
    """Reference ``ParagraphFirstWordCheck``: ``\\n\\n`` paragraphs, nth starts with word."""
    num_paragraphs = _first_present(kw, "num_paragraphs")
    nth = _first_present(kw, "nth_paragraph", "n")
    first_word = str(_one(_first_present(kw, "first_word")) or "").lower()
    if num_paragraphs is None or not nth or not first_word:
        return False
    paragraphs = re.split(r"\n\n", text)
    count = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            count -= 1
    if nth > count:
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    punctuation = {".", ",", "?", "!", "'", '"'}
    first = ""
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    for letter in word:
        if letter in punctuation:
            break
        first += letter.lower()
    return count == num_paragraphs and first == first_word


# ---------------------------------------------------------------------------
# detectable_content:*
# ---------------------------------------------------------------------------


def _check_content_number_placeholders(text: str, kw: dict) -> bool:
    target = _first_present(kw, "num_placeholders", "num")
    if target is None:
        return False
    return len(re.findall(r"\[.*?\]", text)) >= target


def _check_content_postscript(text: str, kw: dict) -> bool:
    marker = str(_one(_first_present(kw, "postscript_marker", "marker")) or "")
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    elif marker:
        pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
    else:
        return False
    return bool(re.findall(pattern, text.lower(), flags=re.MULTILINE))


# ---------------------------------------------------------------------------
# detectable_format:*
# ---------------------------------------------------------------------------


def _check_format_number_bullet_lists(text: str, kw: dict) -> bool:
    target = _first_present(kw, "num_bullets", "num")
    if target is None:
        return False
    bullets = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
    return len(bullets) + len(dashes) == target


def _check_format_constrained_response(text: str, kw: dict) -> bool:
    options = _as_list(_first_present(kw, "options", "choices")) or list(_CONSTRAINED_RESPONSE_OPTIONS)
    value = text.strip()
    return any(str(option) in value for option in options)


def _check_format_number_highlighted_sections(text: str, kw: dict) -> bool:
    target = _first_present(kw, "num_highlights", "num")
    if target is None:
        return False
    n = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", text):
        if highlight.strip("*").strip():
            n += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
        if highlight.removeprefix("**").removesuffix("**").strip():
            n += 1
    return n >= target


def _check_format_multiple_sections(text: str, kw: dict) -> bool:
    splitter = str(_first_present(kw, "section_spliter", "section_splitter", "splitter") or "")
    target = _first_present(kw, "num_sections", "num")
    if not splitter or target is None:
        return False
    sections = re.split(r"\s?" + re.escape(splitter) + r"\s?\d+\s?", text)
    return len(sections) - 1 >= target


def _check_format_json_format(text: str, kw: dict) -> bool:
    value = _strip_code_fence(text)
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _check_format_title(text: str, kw: dict) -> bool:
    for title in re.findall(r"<<[^\n]+>>", text):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def _check_format_sentence_hyphens(text: str, kw: dict) -> bool:
    gold = _split_into_sentences(re.sub("-", " ", text))
    for sentence, gold_sentence in zip(text.split("-"), gold, strict=False):
        if sentence.strip() != sentence:
            return False
        if sentence != gold_sentence:
            return False
    return True


def _check_format_square_brackets(text: str, kw: dict) -> bool:
    return all(word.startswith("[") and word.endswith("]") for word in text.split())


def _check_format_bigram_wrapping(text: str, kw: dict) -> bool:
    words = text.split()
    for i in range(0, len(words) - 1, 2):
        if not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
            return False
    return True


# ---------------------------------------------------------------------------
# paragraphs:*
# ---------------------------------------------------------------------------


def _check_paragraphs_markdown(text: str, kw: dict) -> bool:
    return _count_markdown_paragraphs(text) == 2


def _check_paragraphs_blankline(text: str, kw: dict) -> bool:
    paragraphs = re.split(r"\n\n", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == 2


# ---------------------------------------------------------------------------
# combination:*
# ---------------------------------------------------------------------------


def _check_combination_two_responses(text: str, kw: dict) -> bool:
    valid = []
    responses = text.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _check_combination_repeat_prompt(text: str, kw: dict) -> bool:
    prompt = str(_one(_first_present(kw, "prompt_to_repeat", "prompt")) or "").strip().lower()
    if not prompt:
        return False
    return text.strip().lower().startswith(prompt)


# ---------------------------------------------------------------------------
# copy: / new:*
# ---------------------------------------------------------------------------


def _check_copy_prompt_verbatim(text: str, kw: dict) -> bool:
    prompt = str(_one(_first_present(kw, "prompt_to_repeat", "prompt")) or "")
    if not prompt:
        return False
    return text.strip().lower() == prompt.strip().lower()


def _check_copy_copying_multiple(text: str, kw: dict) -> bool:
    prompt = str(_one(_first_present(kw, "prompt_to_repeat", "prompt")) or "")
    n = _one(_first_present(kw, "N"))
    if not prompt or not n:
        return False
    prompts = text.split("******")
    if len(prompts) != n:
        return False
    return all(p.strip().lower() == prompt.strip().lower() for p in prompts)


def _check_copy_span_idx(text: str, kw: dict) -> bool:
    prompt = str(_one(_first_present(kw, "prompt_to_repeat", "prompt")) or "")
    start = _one(_first_present(kw, "n_start"))
    end = _one(_first_present(kw, "n_end"))
    if not prompt or start is None or end is None:
        return False
    return text.strip().lower() == prompt[start:end].strip().lower()


def _check_copy_repeat_phrase(text: str, kw: dict) -> bool:
    """Reference ``RepeatPhraseChecker`` (keeps upstream's single-total-diff rule)."""
    phrase = str(_one(_first_present(kw, "phrase")) or "").strip()
    small_n = _one(_first_present(kw, "small_n", "N"))
    if not phrase or not small_n:
        return False
    reference = phrase.split()
    found = re.findall(rf"{reference[0]} .*? {reference[-1]}", text)
    if len(found) != small_n:
        return False
    differences = 0
    for candidate in found:
        words = candidate.split()
        if len(words) != len(reference):
            return False
        for i, word in enumerate(words):
            if word != reference[i]:
                differences += 1
                if differences > 1:
                    return False
    return differences == 1


# ---------------------------------------------------------------------------
# startend:*
# ---------------------------------------------------------------------------


def _check_startend_end_checker(text: str, kw: dict) -> bool:
    phrase = str(_one(_first_present(kw, "end_phrase", "phrase")) or "").strip().lower()
    if not phrase:
        return False
    return text.strip().strip('"').lower().endswith(phrase)


def _check_startend_quotation(text: str, kw: dict) -> bool:
    value = text.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


# ---------------------------------------------------------------------------
# change_case:*
# ---------------------------------------------------------------------------


def _check_change_case_capital_word_frequency(text: str, kw: dict) -> bool:
    n = len([word for word in _word_tokenize(text) if word.isupper()])
    relation = _first_present(kw, "capital_relation", "relation")
    return _cmp(n, relation, _first_present(kw, "capital_frequency", "frequency"))


def _check_change_case_english_capital(text: str, kw: dict) -> bool:
    return text.isupper() and _is_english(text)


def _check_change_case_english_lowercase(text: str, kw: dict) -> bool:
    return text.islower() and _is_english(text)


# ---------------------------------------------------------------------------
# punctuation:*
# ---------------------------------------------------------------------------


def _check_punctuation_no_comma(text: str, kw: dict) -> bool:
    return not re.search(r"\,", text)


def _check_punctuation_dot(text: str, kw: dict) -> bool:
    return not re.search(r"\.", text)


def _check_punctuation_exclamation(text: str, kw: dict) -> bool:
    return not re.search(r"\!", text)


# ---------------------------------------------------------------------------
# first_word: / last_word:*
# ---------------------------------------------------------------------------


def _check_first_word_sent(text: str, kw: dict) -> bool:
    expected = str(_one(_first_present(kw, "first_word")) or "").lower()
    if not expected:
        return False
    for sentence in _split_into_sentences(text):
        if not sentence.strip():
            return False
        if sentence.split()[0].strip().lower() != expected:
            return False
    return True


def _check_first_word_answer(text: str, kw: dict) -> bool:
    expected = str(_one(_first_present(kw, "first_word")) or "").lower()
    if not expected or not text.strip() or not text.split():
        return False
    return text.split()[0].strip().lower() == expected


def _check_last_word_sent(text: str, kw: dict) -> bool:
    expected = str(_one(_first_present(kw, "last_word")) or "").lower()
    if not expected:
        return False
    for sentence in _split_into_sentences(text):
        if not sentence.strip():
            return False
        last = re.sub(r"[^\w\s]", "", sentence.split()[-1].strip())
        if last.lower() != expected:
            return False
    return True


def _check_last_word_answer(text: str, kw: dict) -> bool:
    expected = str(_one(_first_present(kw, "last_word")) or "").lower()
    if not expected or not text.split():
        return False
    last = re.sub(r"[^\w\s]", "", text.split()[-1].strip())
    return last.lower() == expected


# ---------------------------------------------------------------------------
# count:*
# ---------------------------------------------------------------------------


def _check_count_lowercase_counting(text: str, kw: dict) -> bool:
    n = _one(_first_present(kw, "N"))
    if n is None:
        return False
    return len(re.findall(r"\b[a-z]+\b", text)) <= n


def _check_count_count_increment_word(text: str, kw: dict) -> bool:
    keyword1 = _one(_first_present(kw, "keyword1"))
    keyword2 = _one(_first_present(kw, "keyword2"))
    if not keyword1 or not keyword2:
        return False
    n1 = len(re.findall(str(keyword1), text, flags=re.IGNORECASE))
    n2 = len(re.findall(str(keyword2), text, flags=re.IGNORECASE))
    return n1 == 1 and n2 == 2


def _check_count_count_unique(text: str, kw: dict) -> bool:
    words = _word_tokenize(text)
    return len(words) == len(set(words))


def _check_count_counting_composition(text: str, kw: dict) -> bool:
    n_sent = _one(_first_present(kw, "n_sent"))
    n_words = _one(_first_present(kw, "n_words"))
    if n_sent is None or n_words is None:
        return False
    paragraphs = _markdown_paragraphs(text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
        sentences = _split_into_sentences(paragraph)
        if len(sentences) != n_sent:
            return False
        for sentence in sentences:
            if len(_word_tokenize(sentence)) != n_words:
                return False
    return count == 3


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def _first_present(kwargs: dict, *names):
    """First non-None value among ``names`` (kwargs spellings vary by task)."""
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return None


_CHECKERS = {
    # keywords
    "keywords:existence": _check_keywords_existence,
    "keywords:frequency": _check_keywords_frequency,
    "keywords:forbidden_words": _check_keywords_forbidden_words,
    "keywords:letter_frequency": _check_keywords_letter_frequency,
    "keywords:word_once": _check_keywords_word_once,
    "keywords:word_count_different_numbers": _check_keywords_word_count_different_numbers,
    "keywords:palindrome": _check_keywords_palindrome,
    "keywords:keyword_specific_position": _check_keywords_keyword_specific_position,
    "keywords:no_adjacent_consecutive": _check_keywords_no_adjacent_consecutive,
    "keywords:start_end": _check_keywords_start_end,
    "keywords:exclude_word_harder": _check_keywords_exclude_word_harder,
    # letters
    "letters:letter_counting": _check_letters_letter_counting,
    "letters:letter_counting2": _check_keywords_letter_frequency,
    # language
    "language:response_language": _check_language_response_language,
    # length
    "length_constraints:number_words": _check_length_number_words,
    "length_constraints:number_sentences": _check_length_number_sentences,
    "length_constraints:number_paragraphs": _check_length_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _check_length_nth_paragraph_first_word,
    # detectable content
    "detectable_content:number_placeholders": _check_content_number_placeholders,
    "detectable_content:postscript": _check_content_postscript,
    # detectable format
    "detectable_format:number_bullet_lists": _check_format_number_bullet_lists,
    "detectable_format:constrained_response": _check_format_constrained_response,
    "detectable_format:number_highlighted_sections": _check_format_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_format_multiple_sections,
    "detectable_format:json_format": _check_format_json_format,
    "detectable_format:title": _check_format_title,
    "detectable_format:sentence_hyphens": _check_format_sentence_hyphens,
    "detectable_format:square_brackets": _check_format_square_brackets,
    "detectable_format:bigram_wrapping": _check_format_bigram_wrapping,
    # paragraphs
    "paragraphs:paragraphs": _check_paragraphs_markdown,
    "paragraphs:paragraphs2": _check_paragraphs_blankline,
    # combination
    "combination:two_responses": _check_combination_two_responses,
    "combination:repeat_prompt": _check_combination_repeat_prompt,
    # copy / new
    "copy:copy": _check_copy_prompt_verbatim,
    "copy:copying_simple": _check_copy_prompt_verbatim,
    "copy:copying_multiple": _check_copy_copying_multiple,
    "copy:repeat_phrase": _check_copy_repeat_phrase,
    "new:copy_span_idx": _check_copy_span_idx,
    # startend
    "startend:end_checker": _check_startend_end_checker,
    "startend:quotation": _check_startend_quotation,
    # change_case
    "change_case:capital_word_frequency": _check_change_case_capital_word_frequency,
    "change_case:english_capital": _check_change_case_english_capital,
    "change_case:english_lowercase": _check_change_case_english_lowercase,
    # punctuation
    "punctuation:no_comma": _check_punctuation_no_comma,
    "punctuation:punctuation_dot": _check_punctuation_dot,
    "punctuation:punctuation_exclamation": _check_punctuation_exclamation,
    # first_word / last_word
    "first_word:first_word_sent": _check_first_word_sent,
    "first_word:first_word_answer": _check_first_word_answer,
    "last_word:last_word_sent": _check_last_word_sent,
    "last_word:last_word_answer": _check_last_word_answer,
    # count
    "count:lowercase_counting": _check_count_lowercase_counting,
    "count:count_increment_word": _check_count_count_increment_word,
    "count:count_unique": _check_count_count_unique,
    "count:counting_composition": _check_count_counting_composition,
}

#: Every instruction id ``check_constraint`` can evaluate.  An id outside this
#: set is graded as failed, so a row carrying one can never score 1.0: the
#: data-level self-check derives its coverage warning from this set.
SUPPORTED_INSTRUCTION_IDS = frozenset(_CHECKERS)


def instruction_ids(constraints: list | None) -> list[str]:
    """Normalise the several constraint spellings to a list of ids."""
    ids = []
    for constraint in constraints or []:
        if isinstance(constraint, str):
            ids.append(constraint)
        elif isinstance(constraint, dict):
            cid = constraint.get("id") or constraint.get("instruction_id") or ""
            ids.append(str(cid))
    return ids


def unsupported_instruction_ids(constraints: list | None) -> list[str]:
    """Ids in ``constraints`` this verifier cannot evaluate (unsorted, deduped)."""
    seen = {}
    for cid in instruction_ids(constraints):
        if cid and cid not in SUPPORTED_INSTRUCTION_IDS:
            seen[cid] = None
    return list(seen)


def check_constraint(constraint: dict, response: str) -> bool:
    """Return True iff the response satisfies one constraint.

    Tolerant of malformed input: unknown ids / missing kwargs -> False.
    """
    if not isinstance(constraint, dict):
        return False
    cid_full = str(constraint.get("id") or constraint.get("instruction_id") or "")
    fn = _CHECKERS.get(cid_full)
    if fn is None:
        return False
    kwargs = constraint.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    try:
        return bool(fn(response, kwargs))
    except Exception:
        return False


def verify_instructions(response: str, constraints: list[dict]) -> tuple[float, list[bool]]:
    """NeMo-Gym convention: reward 1.0 iff EVERY constraint passes.

    Returns (score, per_constraint_follow_list).
    """
    if not constraints:
        return 0.0, []
    follow = [check_constraint(c, response) for c in constraints]
    return (1.0 if all(follow) else 0.0), follow
