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
"""Unit tests for the instruction-following constraint verifier.

Run from the repo root:  pytest examples/reasoning_rl/reward/test_if_verifier.py -v

Every check here is written against the kwargs the real dataset ships (see
``TestRealDatasetKwargs``) and against the reference implementation
``verifiable_instructions`` (github.com/abukharin-nv/verifiable-instructions),
which is what NVIDIA's NeMo-Gym instruction_following environment actually runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from if_verifier import (
    SUPPORTED_INSTRUCTION_IDS,
    check_constraint,
    unsupported_instruction_ids,
    verify_instructions,
)

# Every instruction id that appears in nvidia/Nemotron-RL-instruction_following
# (46,391 rows, 48 distinct ids).  An id missing from SUPPORTED_INSTRUCTION_IDS
# is graded as failed, so a row carrying it can never score 1.0.
DATASET_INSTRUCTION_IDS = frozenset(
    {
        "keywords:existence",
        "keywords:forbidden_words",
        "punctuation:no_comma",
        "detectable_format:title",
        "language:response_language",
        "count:lowercase_counting",
        "detectable_format:bigram_wrapping",
        "punctuation:punctuation_dot",
        "last_word:last_word_answer",
        "detectable_format:sentence_hyphens",
        "punctuation:punctuation_exclamation",
        "detectable_format:number_bullet_lists",
        "startend:end_checker",
        "keywords:word_once",
        "detectable_content:number_placeholders",
        "letters:letter_counting2",
        "first_word:first_word_answer",
        "keywords:letter_frequency",
        "count:count_increment_word",
        "keywords:keyword_specific_position",
        "detectable_format:square_brackets",
        "detectable_format:number_highlighted_sections",
        "copy:repeat_phrase",
        "keywords:word_count_different_numbers",
        "startend:quotation",
        "letters:letter_counting",
        "detectable_content:postscript",
        "last_word:last_word_sent",
        "length_constraints:number_paragraphs",
        "length_constraints:number_words",
        "length_constraints:number_sentences",
        "keywords:palindrome",
        "keywords:start_end",
        "count:count_unique",
        "keywords:no_adjacent_consecutive",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "detectable_format:multiple_sections",
        "keywords:frequency",
        "first_word:first_word_sent",
        "length_constraints:nth_paragraph_first_word",
        "change_case:capital_word_frequency",
        "paragraphs:paragraphs2",
        "paragraphs:paragraphs",
        "count:counting_composition",
        "combination:two_responses",
        "detectable_format:json_format",
        "detectable_format:constrained_response",
    }
)


def follow(cid: str, text: str, **kwargs) -> bool:
    return check_constraint({"id": cid, "kwargs": kwargs or None}, text)


class TestRegistryCoverage:
    def test_every_dataset_id_is_implemented(self):
        missing = DATASET_INSTRUCTION_IDS - SUPPORTED_INSTRUCTION_IDS
        assert not missing, f"dataset uses ids the verifier cannot evaluate: {sorted(missing)}"

    def test_unknown_id_fails_and_is_reported(self):
        constraint = {"id": "made_up:thing", "kwargs": {}}
        assert check_constraint(constraint, "anything") is False
        assert unsupported_instruction_ids([constraint]) == ["made_up:thing"]

    def test_unsupported_instruction_ids_dedupes(self):
        constraints = [
            {"id": "made_up:thing"},
            {"id": "made_up:thing"},
            {"id": "keywords:palindrome", "kwargs": {}},
        ]
        assert unsupported_instruction_ids(constraints) == ["made_up:thing"]


class TestBinaryGrading:
    def test_all_constraints_must_pass(self):
        constraints = [
            {"id": "keywords:palindrome", "kwargs": {}},
            {"id": "punctuation:no_comma", "kwargs": {}},
        ]
        score, follow_list = verify_instructions("radar has no comma", constraints)
        assert score == 1.0 and follow_list == [True, True]
        score, follow_list = verify_instructions("nothing here, radar", constraints)
        assert score == 0.0 and follow_list == [True, False]

    def test_empty_constraints_score_zero(self):
        assert verify_instructions("anything", []) == (0.0, [])


class TestKeywordFamily:
    def test_existence_requires_all_keywords(self):
        kw = {"keywords": ["neat", "spiritual"]}
        assert follow("keywords:existence", "a neat and spiritual answer", **kw)
        assert not follow("keywords:existence", "only neat", **kw)

    def test_frequency_relation(self):
        assert follow("keywords:frequency", "platform platform", keyword="platform", frequency=2, relation="at least")
        assert follow("keywords:frequency", "a platform", keyword="platform", frequency=3, relation="less than")
        assert not follow("keywords:frequency", "p p p", keyword="p", frequency=3, relation="less than")

    def test_forbidden_words_match_whole_words(self):
        kw = {"forbidden_words": ["pass"]}
        assert not follow("keywords:forbidden_words", "you shall pass", **kw)
        # "passing" contains "pass" but is a different word: the reference uses \b...\b
        assert follow("keywords:forbidden_words", "passing by", **kw)

    def test_letter_frequency_uses_let_star_kwargs(self):
        """The dataset ships let_frequency/let_relation, not frequency/relation."""
        kw = {"letter": "e", "let_frequency": 3, "let_relation": "at least"}
        assert follow("keywords:letter_frequency", "eee", **kw)
        assert not follow("keywords:letter_frequency", "ee", **kw)
        kw = {"letter": "c", "let_frequency": 2, "let_relation": "less than"}
        assert follow("keywords:letter_frequency", "abc", **kw)

    def test_letter_counting2_is_the_same_checker(self):
        kw = {"letter": "b", "let_frequency": 5, "let_relation": "less than"}
        assert follow("letters:letter_counting2", "bb", **kw)
        assert not follow("letters:letter_counting2", "bbbbb", **kw)

    def test_word_once(self):
        assert follow("keywords:word_once", "the director arrived", keyword="director")
        assert not follow("keywords:word_once", "the director met the director", keyword="director")

    def test_word_count_different_numbers(self):
        assert follow("keywords:word_count_different_numbers", "rub", keyword="rub", frequency=2, relation="less than")
        assert not follow(
            "keywords:word_count_different_numbers", "rub rub", keyword="rub", frequency=2, relation="less than"
        )

    def test_palindrome(self):
        assert follow("keywords:palindrome", "a level radar")
        assert not follow("keywords:palindrome", "no such thing here")

    def test_keyword_specific_position(self):
        kw = {"keyword": "guess", "n": 2, "m": 2}
        assert follow("keywords:keyword_specific_position", "First one. A guess here.", **kw)
        assert not follow("keywords:keyword_specific_position", "First one. A wild guess.", **kw)

    def test_no_adjacent_consecutive(self):
        assert follow("keywords:no_adjacent_consecutive", "apple cat fish")
        assert not follow("keywords:no_adjacent_consecutive", "apple banana")

    def test_start_end(self):
        assert follow("keywords:start_end", "alpha words alpha")
        assert not follow("keywords:start_end", "alpha words beta")


class TestLengthConstraints:
    def test_number_words(self):
        assert follow("length_constraints:number_words", "one two three", num_words=4, relation="less than")
        assert not follow("length_constraints:number_words", "one two three four", num_words=4, relation="less than")

    def test_number_sentences(self):
        assert follow("length_constraints:number_sentences", "One. Two.", num_sentences=2, relation="at least")
        assert not follow("length_constraints:number_sentences", "One.", num_sentences=2, relation="at least")

    def test_number_paragraphs_uses_markdown_divider(self):
        """The reference splits on '***', not on blank lines."""
        assert follow("length_constraints:number_paragraphs", "One.\n\n***\n\nTwo.", num_paragraphs=2)
        assert not follow("length_constraints:number_paragraphs", "One.\n\nTwo.", num_paragraphs=2)
        assert follow("paragraphs:paragraphs", "One.\n\n***\n\nTwo.")
        assert not follow("paragraphs:paragraphs", "One.\n\n***\n\nTwo.\n\n***\n\nThree.")

    def test_paragraphs2_uses_blank_lines(self):
        assert follow("paragraphs:paragraphs2", "One.\n\nTwo.")
        assert not follow("paragraphs:paragraphs2", "One.\n\n***\n\nTwo.\n\n***\n\nThree.")

    def test_nth_paragraph_first_word_uses_nth_paragraph_kwarg(self):
        kw = {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "blue"}
        assert follow("length_constraints:nth_paragraph_first_word", "First here.\n\nBlue sky above.", **kw)
        assert not follow("length_constraints:nth_paragraph_first_word", "Blue sky.\n\nSecond here.", **kw)
        # num_paragraphs counts paragraphs; it is not the paragraph index.
        assert not follow(
            "length_constraints:nth_paragraph_first_word",
            "First here.",
            num_paragraphs=2,
            nth_paragraph=1,
            first_word="first",
        )


class TestDetectableContent:
    def test_number_placeholders_counts_brackets(self):
        """Regression: the old checker read a `placeholders` kwarg and always passed."""
        assert follow("detectable_content:number_placeholders", "Hello [name]", num_placeholders=1)
        assert not follow("detectable_content:number_placeholders", "Hello there", num_placeholders=1)

    def test_postscript(self):
        assert follow("detectable_content:postscript", "Body.\n\nP.S. thanks", postscript_marker="P.S.")
        assert follow("detectable_content:postscript", "Body.\n\nP.P.S. more", postscript_marker="P.P.S")
        assert not follow("detectable_content:postscript", "Body only.", postscript_marker="P.S.")


class TestDetectableFormat:
    def test_number_bullet_lists_exact_count(self):
        assert follow("detectable_format:number_bullet_lists", "* one\n* two", num_bullets=2)
        # Only markdown * / - bullets count (numbered lists are not bullets).
        assert not follow("detectable_format:number_bullet_lists", "* one\n* two\n- three", num_bullets=2)
        assert follow("detectable_format:number_bullet_lists", "1. one\n2. two", num_bullets=0)

    def test_constrained_response_uses_reference_options(self):
        assert follow("detectable_format:constrained_response", "My answer is yes.")
        assert not follow("detectable_format:constrained_response", "yes")

    def test_number_highlighted_sections(self):
        assert follow("detectable_format:number_highlighted_sections", "a *one* and *two*", num_highlights=2)
        assert not follow("detectable_format:number_highlighted_sections", "a *one*", num_highlights=2)

    def test_multiple_sections(self):
        kw = {"section_spliter": "SECTION", "num_sections": 2}
        assert follow("detectable_format:multiple_sections", "SECTION 1\na\nSECTION 2\nb", **kw)
        assert not follow("detectable_format:multiple_sections", "SECTION 1\na", **kw)

    def test_json_format_accepts_fences(self):
        assert follow("detectable_format:json_format", '```json\n{"a": 1}\n```')
        assert follow("detectable_format:json_format", "[1, 2, 3]")
        assert not follow("detectable_format:json_format", "not json")

    def test_title(self):
        assert follow("detectable_format:title", "<<A Title>>\n\nBody")
        assert not follow("detectable_format:title", "A Title\n\nBody")

    def test_sentence_hyphens(self):
        # The hyphen replaces the whitespace between sentences.
        assert follow("detectable_format:sentence_hyphens", "One two.-Three four.")
        assert not follow("detectable_format:sentence_hyphens", "One sentence here - Another there")

    def test_square_brackets(self):
        assert follow("detectable_format:square_brackets", "[every] [word]")
        assert not follow("detectable_format:square_brackets", "[only] one")

    def test_bigram_wrapping(self):
        assert follow("detectable_format:bigram_wrapping", "<<one two>> <<three four>>")
        assert not follow("detectable_format:bigram_wrapping", "one two three four")


class TestCombinationAndCopy:
    def test_two_responses_must_differ(self):
        assert follow("combination:two_responses", "first\n******\nsecond")
        assert not follow("combination:two_responses", "same\n******\nsame")

    def test_repeat_prompt(self):
        assert follow("combination:repeat_prompt", "What is 2+2? The answer is 4.", prompt_to_repeat="What is 2+2?")
        assert not follow("combination:repeat_prompt", "The answer is 4.", prompt_to_repeat="What is 2+2?")

    def test_copy_verbatim(self):
        assert follow("copy:copy", "Repeat me exactly", prompt_to_repeat="Repeat me exactly")
        assert not follow("copy:copy", "Repeat me", prompt_to_repeat="Repeat me exactly")

    def test_copying_multiple(self):
        kw = {"prompt_to_repeat": "hello", "N": 2}
        assert follow("copy:copying_multiple", "hello\n******\nhello", **kw)
        assert not follow("copy:copying_multiple", "hello", **kw)

    def test_copy_span_idx(self):
        kw = {"prompt_to_repeat": "abcdef", "n_start": 1, "n_end": 4}
        assert follow("new:copy_span_idx", "bcd", **kw)
        assert not follow("new:copy_span_idx", "abc", **kw)

    def test_repeat_phrase(self):
        kw = {"phrase": "The early bird catches the worm", "small_n": 2}
        text = "The early bird catches the worm. The early bird grabs the worm."
        assert follow("copy:repeat_phrase", text, **kw)
        assert not follow("copy:repeat_phrase", "The early bird catches the worm.", **kw)


class TestStartEndCasePunctuation:
    def test_end_checker(self):
        kw = {"end_phrase": "Any other questions?"}
        assert follow("startend:end_checker", "Here you go. Any other questions?", **kw)
        assert not follow("startend:end_checker", "Any other questions? Here you go.", **kw)

    def test_quotation_requires_double_quotes(self):
        assert follow("startend:quotation", '"quoted response"')
        assert not follow("startend:quotation", "'single quoted'")

    def test_capital_word_frequency_uses_capital_kwargs(self):
        """The dataset ships capital_frequency/capital_relation."""
        kw = {"capital_frequency": 2, "capital_relation": "at least"}
        assert follow("change_case:capital_word_frequency", "TWO WORDS here", **kw)
        assert not follow("change_case:capital_word_frequency", "one word here", **kw)

    def test_english_capital_and_lowercase(self):
        assert follow("change_case:english_capital", "ALL CAPS ENGLISH RESPONSE")
        assert not follow("change_case:english_capital", "not all caps")
        assert follow("change_case:english_lowercase", "all lowercase english response")
        assert not follow("change_case:english_lowercase", "Not all lowercase")

    def test_punctuation_checks(self):
        assert follow("punctuation:no_comma", "no commas here")
        assert not follow("punctuation:no_comma", "here, there")
        assert follow("punctuation:punctuation_dot", "no dots here")
        assert not follow("punctuation:punctuation_dot", "a dot.")
        assert follow("punctuation:punctuation_exclamation", "no marks here")
        assert not follow("punctuation:punctuation_exclamation", "wow!")


class TestFirstLastWord:
    def test_first_word_sent(self):
        kw = {"first_word": "start"}
        assert follow("first_word:first_word_sent", "Start here. Start there.", **kw)
        assert not follow("first_word:first_word_sent", "Start here. Then there.", **kw)

    def test_first_word_answer(self):
        assert follow("first_word:first_word_answer", "bike then the rest", first_word="bike")
        assert not follow("first_word:first_word_answer", "the bike", first_word="bike")

    def test_last_word_sent_ignores_trailing_punctuation(self):
        kw = {"last_word": "mud"}
        assert follow("last_word:last_word_sent", "We walked in mud. They played in mud.", **kw)
        assert not follow("last_word:last_word_sent", "We walked in mud. They played outside.", **kw)

    def test_last_word_answer(self):
        assert follow("last_word:last_word_answer", "the response ends with contest", last_word="contest")
        assert not follow("last_word:last_word_answer", "contest first", last_word="contest")


class TestCountFamily:
    def test_lowercase_counting(self):
        assert follow("count:lowercase_counting", "UPPER lower", N=2)
        assert not follow("count:lowercase_counting", "lower lower lower", N=2)

    def test_count_increment_word_unwraps_list_kwargs(self):
        """The dataset stores keyword1/keyword2 as one-element lists."""
        kw = {"keyword1": ["help"], "keyword2": ["dump"]}
        assert follow("count:count_increment_word", "help then dump and dump again", **kw)
        assert not follow("count:count_increment_word", "help then dump", **kw)

    def test_count_unique(self):
        assert follow("count:count_unique", "every word here differs")
        assert not follow("count:count_unique", "same same")

    def test_counting_composition(self):
        # Word counting is NLTK-style: the sentence-final "." is its own token.
        kw = {"n_sent": 2, "n_words": 3}
        sentence = "alpha beta."
        paragraph = f"{sentence} {sentence}"
        assert follow("count:counting_composition", "\n\n***\n\n".join([paragraph] * 3), **kw)
        assert not follow("count:counting_composition", paragraph, **kw)


class TestRealDatasetKwargs:
    """The exact kwargs spellings observed in the published dataset."""

    def test_observed_shapes(self):
        cases = [
            ("keywords:existence", {"keywords": ["neat", "spiritual"]}, "neat and spiritual"),
            ("keywords:forbidden_words", {"forbidden_words": ["inevitable", "part"]}, "clean answer"),
            ("keywords:frequency", {"keyword": "platform", "frequency": 3, "relation": "less than"}, "platform"),
            ("keywords:letter_frequency", {"letter": "e", "let_frequency": 8, "let_relation": "at least"}, "eeeeeeee"),
            ("letters:letter_counting", {"N": 3, "relation": "at least"}, "abcd"),
            ("letters:letter_counting2", {"letter": "c", "let_frequency": 2, "let_relation": "less than"}, "c"),
            (
                "length_constraints:nth_paragraph_first_word",
                {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "crash"},
                "One.\n\nCrash landing here.",
            ),
            ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "One.\n\n***\n\nTwo."),
            ("length_constraints:number_sentences", {"num_sentences": 2, "relation": "less than"}, "One."),
            ("length_constraints:number_words", {"num_words": 3, "relation": "less than"}, "one two"),
            ("detectable_content:number_placeholders", {"num_placeholders": 1}, "[x]"),
            ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body.\n\nP.S. bye"),
            ("detectable_format:number_bullet_lists", {"num_bullets": 1}, "* one"),
            ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* *b*"),
            ("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 1}, "SECTION 1\nx"),
            ("change_case:capital_word_frequency", {"capital_frequency": 1, "capital_relation": "at least"}, "WORD"),
            ("count:lowercase_counting", {"N": 3}, "a b c"),
            ("count:count_increment_word", {"keyword1": ["help"], "keyword2": ["dump"]}, "help dump dump"),
            ("keywords:keyword_specific_position", {"keyword": "guess", "n": 1, "m": 1}, "guess this"),
            (
                "copy:repeat_phrase",
                {"phrase": "Time flies when having fun", "small_n": 2},
                "Time flies when having fun. Time flies while having fun.",
            ),
            ("startend:end_checker", {"end_phrase": "Any other questions?"}, "Sure. Any other questions?"),
            ("detectable_format:constrained_response", None, "My answer is no."),
        ]
        for cid, kw, response in cases:
            assert follow(cid, response, **(kw or {})), f"{cid} should pass for {response!r}"
