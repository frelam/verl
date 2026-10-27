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
"""Unit tests for the stage-4 hard-math adapter (D34).

Everything here is offline: the raw sources are synthetic dicts in the shape of
the HF datasets, so the tests pin down the *build rules* (answer-format filter,
DeepMath difficulty floor, decontamination, dedup, candidate quota, row schema)
without downloading ~160k problems or serving a checkpoint.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hard_math_adapter as adapter
import schema


# ---------------------------------------------------------------------------
# answer normalisation / the rule-friendly filter
# ---------------------------------------------------------------------------


class TestAnswerFilter:
    def test_boxed_and_dollar_wrappers_are_stripped(self):
        assert adapter.normalise_answer("$\\boxed{28}$") == "28"
        assert adapter.normalise_answer("$-4$") == "-4"
        assert adapter.normalise_answer("\\boxed{\\dfrac{1}{2}}") == "\\dfrac{1}{2}"

    def test_integers_and_leading_zeros(self):
        assert adapter.numeric_literal("042") == (Fraction(42), "")
        assert adapter.numeric_literal("1,234") == (Fraction(1234), "")
        assert adapter.numeric_literal("-7") == (Fraction(-7), "")

    def test_simple_fractions_and_decimals(self):
        assert adapter.numeric_literal("1/2") == (Fraction(1, 2), "")
        assert adapter.numeric_literal("\\dfrac{1}{2}") == (Fraction(1, 2), "")
        assert adapter.numeric_literal("\\frac{-3}{4}") == (Fraction(-3, 4), "")
        assert adapter.numeric_literal("0.5") == (Fraction(1, 2), "")

    def test_percent_keeps_its_unit(self):
        assert adapter.numeric_literal("42.86\\%") == (Fraction(2143, 50), "%")

    def test_complex_answers_are_rejected(self):
        for answer in (
            "\\sqrt{2}",
            "2\\pi",
            "[0,1]",
            "(3,4)",
            "A",
            "yes",
            "\\frac{\\sqrt{2}}{2}",
            "x^2 + 1",
            "",
        ):
            assert not adapter.is_rule_friendly_answer(answer), answer

    def test_zero_denominator_rejected(self):
        assert adapter.numeric_literal("1/0") is None


# ---------------------------------------------------------------------------
# source loading / normalisation
# ---------------------------------------------------------------------------


class TestSourceRows:
    def test_deepmath_columns_and_difficulty(self):
        spec = adapter.SOURCES["deepmath"]
        problem, answer, difficulty = adapter._problem_and_answer(
            spec, {"question": "What is 1+1?", "final_answer": "2", "difficulty": 6}
        )
        assert problem == "What is 1+1?"
        assert answer == "2"
        assert difficulty == "6"

    def test_deepscaler_columns(self):
        spec = adapter.SOURCES["deepscaler"]
        problem, answer, _ = adapter._problem_and_answer(
            spec, {"problem": "Find x.", "solution": "...", "answer": "\\frac{1}{2}"}
        )
        assert problem == "Find x."
        assert answer == "\\frac{1}{2}"

    def test_ace_reason_columns(self):
        spec = adapter.SOURCES["ace_reason"]
        problem, answer, _ = adapter._problem_and_answer(
            spec, {"problem": "A question?", "answer": "42"}
        )
        assert problem == "A question?"
        assert answer == "42"

    def test_dapo_verl_native_columns(self):
        spec = adapter.SOURCES["dapo"]
        problem, answer, _ = adapter._problem_and_answer(
            spec,
            {
                "prompt": [{"role": "user", "content": "Solve. " + adapter.MATH_INSTRUCTION}],
                "reward_model": {"ground_truth": "17"},
            },
        )
        assert problem == "Solve."
        assert answer == "17"

    def test_instruction_is_appended_once(self):
        assert adapter._append_instruction("Q?").endswith(adapter.MATH_INSTRUCTION)
        already = "Q? " + adapter.MATH_INSTRUCTION
        assert adapter._append_instruction(already) == already

    def test_image_dependent_problems_are_unusable(self):
        assert not adapter._usable_problem("Look at the figure <image> and find x")
        assert not adapter._usable_problem("[asy] draw((0,0)--(1,1)); [/asy] find the angle")
        assert adapter._usable_problem("A purely textual problem about integers, long enough.")


class TestCollectCandidates:
    RAW = {
        "ace_reason": [
            {"problem": "Ace problem one with enough text?", "answer": "12"},
            {"problem": "Ace problem two with enough text?", "answer": "\\sqrt{2}"},
            {"problem": "Ace problem three with enough text?", "answer": "$\\boxed{7}$"},
        ],
        "deepscaler": [
            {"problem": "DeepScaleR problem one with text?", "answer": "3/4"},
        ],
        "deepmath": [
            {"question": "DeepMath easy problem with enough text?", "final_answer": "5", "difficulty": 4},
            {"question": "DeepMath hard problem with enough text?", "final_answer": "6", "difficulty": 7},
            {"question": "DeepMath symbolic hard problem with text?", "final_answer": "2\\pi", "difficulty": 9},
        ],
    }

    def _collect(self, monkeypatch, keep_symbolic=False):
        def fake_load(source, raw_dir):
            return self.RAW[source.key]

        monkeypatch.setattr(adapter, "load_raw_rows", fake_load)
        return adapter.collect_candidates(
            "/tmp/does-not-matter",
            ["ace_reason", "deepscaler", "deepmath"],
            deepmath_min_difficulty=6,
            keep_symbolic=keep_symbolic,
        )

    def test_funnel_applies_answer_and_difficulty_filters(self, monkeypatch):
        candidates, funnel = self._collect(monkeypatch)
        # ace: 12 and \boxed{7} survive, \sqrt{2} does not
        assert funnel["ace_reason"]["raw"] == 3
        assert funnel["ace_reason"]["admissible"] == 2
        assert funnel["ace_reason"]["non_rule_answer"] == 1
        # deepscaler: 3/4 survives
        assert funnel["deepscaler"]["admissible"] == 1
        # deepmath: level 4 is below the floor, level 7 survives, level 9 is
        # symbolic and dropped -> 1 admissible out of 3
        assert funnel["deepmath"]["below_min_difficulty"] == 1
        assert funnel["deepmath"]["non_rule_answer"] == 1
        assert funnel["deepmath"]["admissible"] == 1
        assert len(candidates) == 4
        assert {c["data_source"] for c in candidates} == {
            "math_hard_ace_reason",
            "math_hard_deepscaler",
            "math_hard_deepmath",
        }

    def test_keep_symbolic_keeps_the_complex_answers(self, monkeypatch):
        candidates, funnel = self._collect(monkeypatch, keep_symbolic=True)
        assert funnel["deepmath"].get("non_rule_answer", 0) == 0
        assert funnel["ace_reason"].get("non_rule_answer", 0) == 0
        assert len(candidates) == 6

    def test_task_ids_are_unique_across_sources(self, monkeypatch):
        candidates, _ = self._collect(monkeypatch)
        ids = [candidate["task_id"] for candidate in candidates]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# decontamination / dedup
# ---------------------------------------------------------------------------


class TestDecontamination:
    def test_ngram_overlap_removes_a_held_out_problem(self):
        held_out = [
            "Let S be the set of all integers n such that n squared plus one divides 2024."
        ]
        rows = [
            {"_problem": held_out[0]},
            {"_problem": "A completely different problem about triangles and their angles."},
        ]
        kept, report = adapter.decontaminate(
            rows, held_out, ngram_n=8, minhash_threshold=0.0
        )
        assert len(kept) == 1
        assert report["ngram_dropped"] == 1

    def test_no_held_out_text_is_a_noop(self):
        rows = [{"_problem": "anything at all"}]
        kept, report = adapter.decontaminate(rows, [], ngram_n=8, minhash_threshold=0.8)
        assert kept == rows
        assert report["ngram_dropped"] == 0

    def test_near_verbatim_copy_is_removed(self):
        held_out = ["Find the number of ordered pairs of positive integers (a, b) with a + b = 100."]
        rows = [{"_problem": "Find the number of ordered pairs of positive integers (a,b) with a+b=100 ."}]
        kept, report = adapter.decontaminate(rows, held_out, ngram_n=8, minhash_threshold=0.8)
        assert not kept
        assert report["ngram_dropped"] + report["minhash_dropped"] == 1


class TestDedup:
    def test_exact_duplicate_is_dropped(self):
        rows = [
            {"_problem": "The same problem statement, word for word."},
            {"_problem": "The  same problem statement, word for word."},
            {"_problem": "A different problem statement entirely."},
        ]
        kept, report = adapter.dedup_pool(rows, threshold=0.0)
        assert len(kept) == 2
        assert report["exact_dropped"] == 1

    def test_near_duplicate_is_dropped(self):
        rows = [
            {"_problem": "Find all integers n such that n^2 - 3n + 2 divides 2024 exactly."},
            {"_problem": "Find all integers n such that n^2 - 3n + 2 divides 2024 exact."},
        ]
        kept, report = adapter.dedup_pool(rows, threshold=0.6)
        assert report["near_dropped"] == 1
        assert len(kept) == 1

    def test_anchor_dedup_only_drops_the_query_side(self):
        rows = [{"_problem": "An old problem that was already trained on."}]
        anchors = ["An old problem that was already trained on."]
        kept, dropped = adapter.dedup_against_anchors(rows, anchors, threshold=0.6)
        assert dropped == 1
        assert not kept

    def test_anchor_dedup_catches_a_reworded_copy(self):
        anchors = ["Find all integers n such that n squared minus three n plus two divides 2024."]
        rows = [{"_problem": "Find all integers n such that n squared minus three n plus two divides 2025."}]
        kept, dropped = adapter.dedup_against_anchors(rows, anchors, threshold=0.6)
        assert dropped == 1
        assert not kept


class TestSemanticDedup:
    """Embedding-level dedup (the 'semantic similarity' half of the plan)."""

    def _rows(self, vectors):
        return [{"_problem": f"problem {i}"} for i in range(len(vectors))]

    def test_disabled_without_an_encoder(self):
        rows = [{"_problem": "a"}, {"_problem": "b"}]
        kept, report = adapter.semantic_dedup(rows, None, threshold=0.95)
        assert kept == rows
        assert report["enabled"] is False

    def test_similar_vectors_are_dropped(self):
        def encode(_texts):
            return [
                [1.0, 0.0],  # A kept
                [0.999, 0.04],  # B ~ A -> dropped
                [0.0, 1.0],  # C kept
                [0.04, 0.999],  # D ~ C -> dropped
                [0.7, 0.7],  # E unrelated enough -> kept
                [0.71, 0.71],  # F ~ E -> dropped
            ]

        rows = self._rows(range(6))
        kept, report = adapter.semantic_dedup(rows, encode, threshold=0.99)
        assert [row["_problem"] for row in kept] == ["problem 0", "problem 2", "problem 4"]
        assert report["enabled"] is True
        assert report["dropped"] == 3

    def test_identical_rows_are_always_found_by_lsh(self):
        import numpy as np

        rng = np.random.default_rng(7)
        base = rng.standard_normal((100, 32)).astype("float32")
        vectors = np.vstack([base, base[0:1]])  # row 100 duplicates row 0

        def encode(_texts):
            return vectors

        rows = self._rows(range(101))
        kept, report = adapter.semantic_dedup(rows, encode, threshold=0.99)
        assert len(kept) == 100
        assert all(row["_problem"] != "problem 100" for row in kept)
        assert report["lsh_bits"] == 64
        assert report["lsh_bands"] == 8

    def test_small_signature_is_clamped_to_whole_bands(self):
        def encode(_texts):
            return [[1.0, 0.0], [0.0, 1.0]]

        kept, report = adapter.semantic_dedup(
            self._rows(range(2)), encode, threshold=0.99, bits=4, bands=9
        )
        assert len(kept) == 2
        assert report["lsh_bits"] == 8
        assert report["lsh_bands"] == 1

    def test_encoder_shape_mismatch_is_rejected(self):
        def encode(_texts):
            return [[1.0, 0.0]]

        with pytest.raises(ValueError, match="expected"):
            adapter.semantic_dedup(self._rows(range(2)), encode, threshold=0.95)

    def test_non_positive_threshold_disables_the_layer(self):
        rows = self._rows(range(2))
        kept, report = adapter.semantic_dedup(rows, lambda texts: [[1.0, 0.0]] * len(texts), threshold=0.0)
        assert kept == rows
        assert report["enabled"] is False


class TestCandidateQuota:
    def test_quota_is_split_by_weight_and_capped_by_supply(self):
        rows = (
            [{"_source": "a"}] * 100
            + [{"_source": "b"}] * 10
            + [{"_source": "c"}] * 100
        )
        chosen, report = adapter.apply_candidate_quota(
            rows, {"a": 0.5, "b": 0.25, "c": 0.25}, quota=80, seed=0
        )
        assert len(chosen) == 80
        # b cannot supply its 20 rows, so the remainder stays inside a/c.
        assert report["allocation"]["b"] == 10
        assert report["allocation"]["a"] + report["allocation"]["c"] == 70

    def test_zero_quota_keeps_everything(self):
        rows = [{"_source": "a"}] * 5
        chosen, report = adapter.apply_candidate_quota(rows, {"a": 1.0}, quota=0, seed=0)
        assert len(chosen) == 5
        assert report["allocation"] == {"a": 5}


class TestWeights:
    def test_short_names_and_normalisation(self):
        weights = adapter.parse_weights("ace=2,deepscaler=1,deepmath=1")
        assert weights["ace_reason"] == pytest.approx(0.5)
        assert weights["deepscaler"] == pytest.approx(0.25)
        assert weights["dapo"] == pytest.approx(0.0)

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="unknown --weights"):
            adapter.parse_weights("nope=1.0")


# ---------------------------------------------------------------------------
# row schema
# ---------------------------------------------------------------------------


class TestBuildRows:
    def test_rows_match_the_stage1_schema(self):
        candidates = [
            {
                "_source": "deepmath",
                "_problem": "A hard problem with enough text.",
                "_answer": "42",
                "_difficulty": "7",
                "_raw_index": 3,
                "prompt": [{"role": "user", "content": "A hard problem with enough text. " + adapter.MATH_INSTRUCTION}],
                "answer": "42",
                "data_source": "math_hard_deepmath",
                "task_id": "hardmath-deepmath-3",
                "pass_rate": 0.375,
            }
        ]
        rows = adapter.build_rows(candidates, seed=44)
        assert len(rows) == 1
        row = rows[0]
        assert row["data_source"] == "math_hard_deepmath"
        assert row["ability"] == "math"
        assert row["reward_model"]["ground_truth"] == "42"
        assert row["extra_info"]["domain"] == "math"
        assert row["extra_info"]["difficulty"] == "7"
        assert row["extra_info"]["pass_rate"] == pytest.approx(0.375)
        assert row["extra_info"]["task_id"] == "hardmath-deepmath-3"
        # Every canonical extra_info key is present, so the mixed Arrow struct
        # cannot be widened by this domain at write time.
        assert set(schema.EXTRA_INFO_KEYS) <= set(row["extra_info"])

    def test_unmeasured_rows_keep_the_unset_sentinel(self):
        candidates = [
            {
                "_source": "ace_reason",
                "_problem": "Another hard problem with enough text.",
                "_answer": "1/2",
                "_difficulty": "",
                "_raw_index": 0,
                "prompt": [{"role": "user", "content": "Another hard problem with enough text."}],
                "answer": "1/2",
                "data_source": "math_hard_ace_reason",
                "task_id": "hardmath-ace_reason-0",
            }
        ]
        row = adapter.build_rows(candidates, seed=44)[0]
        assert row["extra_info"]["pass_rate"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# difficulty-filter preflight
# ---------------------------------------------------------------------------


class TestRewardPreflight:
    def test_a_broken_math_reward_is_rejected(self):
        # A reward that returns 0 for a correct AIME-style answer would mark the
        # whole pool all-wrong; the preflight must stop the build instead.
        def broken_reward(**kwargs):
            return {"score": 0.0}

        with pytest.raises(RuntimeError, match="preflight"):
            adapter.preflight_reward(broken_reward)

    def test_a_working_reward_passes(self):
        def working_reward(*, solution_str, ground_truth, **kwargs):
            expected = adapter.numeric_literal(ground_truth)
            boxed = solution_str.split("\\boxed{")[-1].split("}")[0]
            return {"score": 1.0 if adapter.numeric_literal(boxed) == expected else 0.0}

        adapter.preflight_reward(working_reward)


# ---------------------------------------------------------------------------
# CLI guards
# ---------------------------------------------------------------------------


class TestMainGuards:
    def test_decontamination_is_mandatory(self, tmp_path):
        out = tmp_path / "hard_math.parquet"
        with pytest.raises(ValueError, match="decontaminate-path"):
            adapter.main(["--out", str(out), "--skip-difficulty-filter"])

    def test_bad_pass_range_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="min-pass"):
            adapter.main([
                "--out", str(tmp_path / "hard_math.parquet"),
                "--allow-no-decontamination",
                "--skip-difficulty-filter",
                "--min-pass", "0.9",
                "--max-pass", "0.1",
            ])

    def test_zero_weight_source_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="zero --weights share"):
            adapter.main([
                "--out", str(tmp_path / "hard_math.parquet"),
                "--allow-no-decontamination",
                "--skip-difficulty-filter",
                "--sources", "deepmath",
                "--weights", "deepmath=0",
            ])
