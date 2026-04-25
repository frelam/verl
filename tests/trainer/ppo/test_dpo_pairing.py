# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for compute_dpo_preferences (best_vs_worst pairing)."""

import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_dpo_preferences


def _make_data_proto(num_groups=3, n_per_group=4, seq_len=6, seed=0):
    """Build a DataProto with ``num_groups`` prompt groups, ``n_per_group`` samples each.

    Scores are set so that within each group they are distinct (0.1, 0.2, ...).
    The reward tensor places the scalar score at the last valid token position.
    """
    torch.manual_seed(seed)
    bsz = num_groups * n_per_group
    # Token-level scores: zeros except last valid position holds the scalar score
    token_scores = torch.zeros(bsz, seq_len)
    response_mask = torch.ones(bsz, seq_len, dtype=torch.float32)
    # Mask last column to ensure scoring uses mask correctly
    response_mask[:, -1] = 0

    scores_per_sample = []
    uids = []
    for g in range(num_groups):
        for i in range(n_per_group):
            score = 0.1 * (i + 1)  # 0.1, 0.2, 0.3, 0.4
            scores_per_sample.append(score)
            uids.append(f"prompt_{g}")

    # Place score at the last *valid* token (index seq_len-2 since -1 is masked)
    for idx, score in enumerate(scores_per_sample):
        token_scores[idx, seq_len - 2] = score

    batch = TensorDict(
        source={
            "token_level_scores": token_scores,
            "response_mask": response_mask,
            "responses": torch.randint(0, 100, (bsz, seq_len)),
            "old_log_probs": torch.randn(bsz, seq_len),
        },
        batch_size=(bsz,),
    )
    non_tensor_batch = {
        "uid": np.array(uids),
        "data_source": np.array(["test"] * bsz),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class TestComputeDpoPreferences(unittest.TestCase):
    def test_best_vs_worst_pairing(self):
        """Each group should produce exactly 1 pair (best vs worst)."""
        data = _make_data_proto(num_groups=3, n_per_group=4, seq_len=6)
        result = compute_dpo_preferences(data, reward_key="token_level_scores")
        # 3 groups -> 3 pairs -> 6 samples (interleaved)
        self.assertEqual(len(result), 6)

    def test_pairing_interleave_order(self):
        """Output should be [c0, r0, c1, r1, ...]."""
        data = _make_data_proto(num_groups=2, n_per_group=4, seq_len=6)
        result = compute_dpo_preferences(data, reward_key="token_level_scores")
        # Verify by checking the scores: chosen (even idx) should be 0.4 (max),
        # rejected (odd idx) should be 0.1 (min)
        token_scores = result.batch["token_level_scores"]
        response_mask = result.batch["response_mask"]
        scores = (token_scores * response_mask).sum(dim=-1)
        # Pair 0: chosen=0.4, rejected=0.1
        self.assertAlmostEqual(scores[0].item(), 0.4, places=5)
        self.assertAlmostEqual(scores[1].item(), 0.1, places=5)
        # Pair 1: chosen=0.4, rejected=0.1
        self.assertAlmostEqual(scores[2].item(), 0.4, places=5)
        self.assertAlmostEqual(scores[3].item(), 0.1, places=5)

    def test_preference_label(self):
        """preference_label should be [1, 0, 1, 0, ...]."""
        data = _make_data_proto(num_groups=2, n_per_group=3, seq_len=5)
        result = compute_dpo_preferences(data, reward_key="token_level_scores")
        label = result.batch["preference_label"]
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.assertTrue(torch.equal(label, expected))

    def test_uid_preserved(self):
        """Each pair should carry the correct uid from its group."""
        data = _make_data_proto(num_groups=2, n_per_group=3, seq_len=5)
        result = compute_dpo_preferences(data, reward_key="token_level_scores")
        uids = result.non_tensor_batch["uid"]
        # Pair 0 from prompt_0, pair 1 from prompt_1
        self.assertEqual(uids[0], "prompt_0")
        self.assertEqual(uids[1], "prompt_0")
        self.assertEqual(uids[2], "prompt_1")
        self.assertEqual(uids[3], "prompt_1")

    def test_skip_tied_scores(self):
        """Groups where all scores are equal should be skipped (no pair)."""
        bsz = 4
        seq_len = 5
        token_scores = torch.zeros(bsz, seq_len)
        # All scores = 0.5 -> tie -> skipped
        token_scores[:, seq_len - 2] = 0.5
        response_mask = torch.ones(bsz, seq_len)
        response_mask[:, -1] = 0
        batch = TensorDict(
            source={
                "token_level_scores": token_scores,
                "response_mask": response_mask,
                "responses": torch.zeros(bsz, seq_len, dtype=torch.long),
                "old_log_probs": torch.zeros(bsz, seq_len),
            },
            batch_size=(bsz,),
        )
        non_tensor_batch = {"uid": np.array(["p0"] * bsz)}
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        with self.assertRaises(ValueError):
            compute_dpo_preferences(data, reward_key="token_level_scores")

    def test_skip_single_sample_groups(self):
        """Groups with <2 samples cannot form a pair and should be skipped."""
        bsz = 2
        seq_len = 4
        token_scores = torch.zeros(bsz, seq_len)
        token_scores[0, -2] = 0.3
        token_scores[1, -2] = 0.7
        response_mask = torch.ones(bsz, seq_len)
        response_mask[:, -1] = 0
        batch = TensorDict(
            source={
                "token_level_scores": token_scores,
                "response_mask": response_mask,
                "responses": torch.zeros(bsz, seq_len, dtype=torch.long),
                "old_log_probs": torch.zeros(bsz, seq_len),
            },
            batch_size=(bsz,),
        )
        # Each sample is its own group (single sample) -> no pairs
        non_tensor_batch = {"uid": np.array(["p0", "p1"])}
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        with self.assertRaises(ValueError):
            compute_dpo_preferences(data, reward_key="token_level_scores")


if __name__ == "__main__":
    unittest.main()
