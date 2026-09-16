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
"""Unit tests for the Reasoning Gym official-verifier bridge.

Run from the repo root:  pytest examples/reasoning_rl/reward/test_reasoning_gym_verifier.py -v

The plumbing tests use a fake dataset, so they run without ``reasoning-gym``
installed; the integration tests (which need the real library) are skipped when
the package is missing.
"""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_gym_verifier import score_with_dataset, seed_from_extra_info


class _FakeDataset:
    """Minimal stand-in for a reasoning_gym dataset (item = answers[seed + idx])."""

    def __init__(self, answers, full_credit=(), raise_on_index=False):
        self.answers = list(answers)
        self.full_credit = set(full_credit)
        self.raise_on_index = raise_on_index
        self.seed = 0
        self.calls = []

    def __getitem__(self, idx):
        if self.raise_on_index:
            raise RuntimeError("generator exploded")
        return {"answer": self.answers[self.seed + idx], "question": f"q{self.seed + idx}"}

    def score_answer(self, answer, entry):
        self.calls.append((answer, entry["answer"]))
        return 1.0 if answer in self.full_credit else 0.0


# ---------------------------------------------------------------------------
# seed plumbing
# ---------------------------------------------------------------------------


class TestSeedFromExtraInfo:
    def test_seed_field(self):
        assert seed_from_extra_info({"seed": 123}) == 123

    def test_numeric_string_seed(self):
        assert seed_from_extra_info({"seed": "17"}) == 17

    def test_task_id_fallback(self):
        # Rows written before extra_info.seed existed still carry rgym-<task>-<seed>.
        assert seed_from_extra_info({"task_id": "rgym-word_ladder-4096"}) == 4096

    def test_task_id_with_dashes_in_task(self):
        assert seed_from_extra_info({"task_id": "rgym-game-of-life-8"}) == 8

    def test_missing_seed(self):
        assert seed_from_extra_info({"task_id": "enigmata-maze-3"}) is None
        assert seed_from_extra_info({}) is None

    def test_non_dict(self):
        assert seed_from_extra_info(None) is None
        assert seed_from_extra_info("seed=3") is None

    def test_unusable_seed_falls_through_to_task_id(self):
        assert seed_from_extra_info({"seed": "n/a", "task_id": "rgym-countdown-5"}) == 5


# ---------------------------------------------------------------------------
# scoring core
# ---------------------------------------------------------------------------


class TestScoreWithDataset:
    def test_alternative_answer_accepted_by_task_verifier(self):
        ds = _FakeDataset(answers=["canonical", "other"], full_credit={"equivalent"})
        assert score_with_dataset(ds, "equivalent", "canonical", 0) is True
        assert ds.calls == [("equivalent", "canonical")]

    def test_seed_selects_the_matching_entry(self):
        ds = _FakeDataset(answers=["a0", "a1", "a2", "a3"])
        assert score_with_dataset(ds, "whatever", "a2", 2) is False  # correct entry, no credit
        assert ds.calls == [("whatever", "a2")]

    def test_partial_credit_is_not_credit(self):
        class _Partial(_FakeDataset):
            def score_answer(self, answer, entry):
                return 0.5

        assert score_with_dataset(_Partial(["a"]), "a", "a", 0) is False

    def test_none_prediction_unhandled(self):
        ds = _FakeDataset(answers=["a"])
        assert score_with_dataset(ds, None, "a", 0) is None
        assert ds.calls == []

    def test_missing_seed_unhandled(self):
        ds = _FakeDataset(answers=["a"])
        assert score_with_dataset(ds, "a", "a", None) is None
        assert score_with_dataset(ds, "a", "a", "not-a-number") is None
        assert ds.calls == []

    def test_no_seed_sentinel_unhandled(self):
        assert score_with_dataset(_FakeDataset(["a"]), "a", "a", -1) is None

    def test_ground_truth_drift_refuses_to_score(self):
        # A version bump (or a foreign seed) must degrade to string comparison,
        # never to "credit whatever the regenerated puzzle says".
        ds = _FakeDataset(answers=["regenerated"], full_credit={"model answer"})
        assert score_with_dataset(ds, "model answer", "stored answer", 0) is None
        assert ds.calls == []

    def test_regeneration_failure_unhandled(self):
        ds = _FakeDataset(answers=["a"], raise_on_index=True)
        assert score_with_dataset(ds, "a", "a", 0) is None

    def test_verifier_exception_unhandled(self):
        class _Boom(_FakeDataset):
            def score_answer(self, answer, entry):
                raise ValueError("bad expression")

        assert score_with_dataset(_Boom(["a"]), "a", "a", 0) is None


# ---------------------------------------------------------------------------
# integration (needs the real reasoning-gym package)
# ---------------------------------------------------------------------------


@pytest.fixture
def reasoning_gym():
    return pytest.importorskip("reasoning_gym")


def _row(reasoning_gym, task, seed=0):
    dataset = reasoning_gym.create_dataset(task, size=1, seed=0)
    dataset.seed = seed
    entry = dataset[0]
    payload = json.dumps({"answer": str(entry["answer"]).strip(), "task": task})
    return entry, payload


class TestReasoningGymIntegration:
    def test_regenerated_entry_matches_batch_index(self, reasoning_gym):
        # The bridge relies on item = Random(seed + idx); a batch entry must be
        # reproducible from its seed alone.
        batch = reasoning_gym.create_dataset("countdown", size=4, seed=0)
        for seed in range(4):
            _, payload = _row(reasoning_gym, "countdown", seed)
            assert json.loads(payload)["answer"] == str(batch[seed]["answer"]).strip()

    def test_countdown_equivalent_expression_credited(self, reasoning_gym):
        from compute_score import compute_score

        entry, payload = _row(reasoning_gym, "countdown", 0)
        compact = re.sub(r"\s+", "", entry["answer"])
        assert compact != entry["answer"], "fixture expression is already space-free"
        response = f"<think>x</think>\n\n<answer>{compact}</answer>"

        # String comparison cannot see the equivalence; the library verifier can.
        assert compute_score("logic_reasoning_gym", response, payload)["score"] == 0.0
        assert compute_score("logic_reasoning_gym", response, payload, extra_info={"seed": 0})["score"] == 1.0

    def test_wrong_answer_still_zero(self, reasoning_gym):
        from compute_score import compute_score

        _, payload = _row(reasoning_gym, "countdown", 0)
        response = "<think>x</think>\n\n<answer>1 + 1</answer>"
        assert compute_score("logic_reasoning_gym", response, payload, extra_info={"seed": 0})["score"] == 0.0

    def test_unique_answer_tasks_keep_string_match_recall(self, reasoning_gym):
        # Our matcher is deliberately more lenient than the library on sudoku
        # layout; the bridge must only ever ADD credit.
        from compute_score import compute_score

        entry, payload = _row(reasoning_gym, "sudoku", 0)
        flattened = " ".join(str(entry["answer"]).split("\n"))
        assert flattened != entry["answer"]
        response = f"<think>x</think>\n\n<answer>{flattened}</answer>"
        assert compute_score("logic_reasoning_gym", response, payload, extra_info={"seed": 0})["score"] == 1.0

    def test_bare_answer_response_is_extracted(self, reasoning_gym):
        # reasoning-gym prompts say "respond with only your answer", so a
        # compliant response often carries no <answer> tag.
        from compute_score import compute_score

        entry, payload = _row(reasoning_gym, "maze", 0)
        response = f"<think>the shortest route is {entry['answer']} steps</think>\n\n{entry['answer']}"
        assert compute_score("logic_reasoning_gym", response, payload, extra_info={"seed": 0})["score"] == 1.0
