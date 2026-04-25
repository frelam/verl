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

"""Unit tests for the DPO policy loss (compute_policy_loss_dpo)."""

import math
import unittest
from dataclasses import dataclass, field
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_dpo, get_policy_loss_fn


@dataclass
class _MockPolicyLossConfig:
    dpo_beta: float = 0.1


@dataclass
class _MockActorConfig:
    policy_loss: _MockPolicyLossConfig = field(default_factory=_MockPolicyLossConfig)
    batch_context: dict = field(default_factory=dict)


class TestDPOLoss(unittest.TestCase):
    def _make_inputs(self, num_pairs=3, seq_len=5, beta=0.1, seed=0):
        """Create synthetic inputs in interleave order [c0, r0, c1, r1, ...]."""
        torch.manual_seed(seed)
        bsz = 2 * num_pairs
        log_prob = torch.randn(bsz, seq_len, requires_grad=True)
        ref_log_prob = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        # Make some tokens masked to verify masking works
        response_mask[:, -1] = 0
        preference_label = torch.zeros(bsz)
        preference_label[0::2] = 1.0  # chosen
        preference_label[1::2] = 0.0  # rejected
        old_log_prob = torch.randn(bsz, seq_len)
        advantages = torch.zeros(bsz, seq_len)
        config = _MockActorConfig()
        config.batch_context = {
            "ref_log_prob": ref_log_prob,
            "preference_label": preference_label,
        }
        return log_prob, ref_log_prob, response_mask, preference_label, old_log_prob, advantages, config

    def test_dpo_loss_registered(self):
        """The 'dpo' loss should be in the registry."""
        fn = get_policy_loss_fn("dpo")
        self.assertIs(fn, compute_policy_loss_dpo)

    def test_dpo_loss_correctness(self):
        """Loss equals -log σ(β * (chosen_sum - rejected_sum))."""
        log_prob, ref_log_prob, response_mask, preference_label, old_log_prob, advantages, config = (
            self._make_inputs(num_pairs=4, seq_len=6, beta=0.2)
        )
        config.policy_loss.dpo_beta = 0.2

        loss, metrics = compute_policy_loss_dpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
        )

        # Manual computation
        pi_logratio = log_prob - ref_log_prob
        seq_logratio = (pi_logratio * response_mask).sum(dim=-1)
        chosen = seq_logratio[0::2]
        rejected = seq_logratio[1::2]
        logits = 0.2 * (chosen - rejected)
        expected_loss = -torch.nn.functional.logsigmoid(logits).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    def test_dpo_gradient_flow(self):
        """Loss should have gradients flowing to log_prob."""
        log_prob, ref_log_prob, response_mask, preference_label, old_log_prob, advantages, config = (
            self._make_inputs(num_pairs=2, seq_len=4)
        )
        loss, _ = compute_policy_loss_dpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
        )
        loss.backward()
        self.assertIsNotNone(log_prob.grad)
        self.assertTrue(torch.any(log_prob.grad != 0))

    def test_dpo_metrics_keys(self):
        """All expected DPO metrics should be present."""
        log_prob, ref_log_prob, response_mask, preference_label, old_log_prob, advantages, config = (
            self._make_inputs(num_pairs=2, seq_len=4)
        )
        _, metrics = compute_policy_loss_dpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
        )
        for key in [
            "actor/dpo_loss",
            "actor/dpo_accuracy",
            "actor/dpo_margin",
            "actor/dpo_chosen_reward",
            "actor/dpo_rejected_reward",
        ]:
            self.assertIn(key, metrics, f"Missing metric: {key}")

    def test_dpo_accuracy_when_chosen_better(self):
        """When chosen has higher log-ratio than rejected, accuracy should be 1.0."""
        seq_len = 4
        # chosen: log_prob=0, ref_log_prob=-1 => logratio=+1
        # rejected: log_prob=0, ref_log_prob=+1 => logratio=-1
        log_prob = torch.zeros(2, seq_len)
        ref_log_prob = torch.tensor([[-1.0] * seq_len, [1.0] * seq_len])
        response_mask = torch.ones(2, seq_len)
        preference_label = torch.tensor([1.0, 0.0])
        config = _MockActorConfig()
        config.batch_context = {"ref_log_prob": ref_log_prob, "preference_label": preference_label}
        _, metrics = compute_policy_loss_dpo(
            old_log_prob=torch.zeros(2, seq_len),
            log_prob=log_prob,
            advantages=torch.zeros(2, seq_len),
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
        )
        self.assertAlmostEqual(metrics["actor/dpo_accuracy"], 1.0, places=5)
        # Margin = chosen_sum - rejected_sum
        # chosen: log_prob=0, ref_log_prob=-1 => per-token logratio=+1, sum over 4 tokens = +4
        # rejected: log_prob=0, ref_log_prob=+1 => per-token logratio=-1, sum over 4 tokens = -4
        # margin = 4 - (-4) = 8
        self.assertAlmostEqual(metrics["actor/dpo_margin"], 8.0, places=5)

    def test_dpo_batch_interleave_order(self):
        """Verify the loss correctly pairs even=chosen, odd=rejected indices."""
        seq_len = 3
        # 2 pairs: [c0, r0, c1, r1]
        log_prob = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        # Make c0 better than r0, c1 worse than r1
        ref_log_prob = torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
        )
        response_mask = torch.ones(4, seq_len)
        preference_label = torch.tensor([1.0, 0.0, 1.0, 0.0])
        config = _MockActorConfig()
        config.batch_context = {"ref_log_prob": ref_log_prob, "preference_label": preference_label}
        _, metrics = compute_policy_loss_dpo(
            old_log_prob=torch.zeros(4, seq_len),
            log_prob=log_prob,
            advantages=torch.zeros(4, seq_len),
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
        )
        # pair0: chosen_logratio=+3, rejected_logratio=-3 => correct
        # pair1: chosen_logratio=-3, rejected_logratio=+3 => incorrect
        # accuracy = 0.5
        self.assertAlmostEqual(metrics["actor/dpo_accuracy"], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
