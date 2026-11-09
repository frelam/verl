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
"""Lightweight tests for selection-only built-in replay-buffer samplers."""

import importlib.util
from pathlib import Path

import pytest


def _load_samplers_module():
    # Import the standalone module directly so CPU tests do not require torch or TransferQueue.
    module_path = Path(__file__).resolve().parents[4] / "verl" / "trainer" / "ppo" / "v1" / "samplers.py"
    spec = importlib.util.spec_from_file_location("v1_builtin_samplers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


samplers = _load_samplers_module()


class _FakeBufferBase:
    """Minimal base contract used by dynamically composed selection mixins."""

    def __init__(self, prompt_steps, rewards=None, lengths=None, sampler_kwargs=None):
        self.partitions = {"train": {}}
        self.prompt_global_steps = {"train": dict(prompt_steps)}
        self.sampler_kwargs = sampler_kwargs or {}
        self.rewards = rewards or {}
        self.reward_calls = []
        self._dapo_classification_cache = {"train": {}}

        lengths = lengths or {}
        for uid in prompt_steps:
            group_lengths = lengths.get(uid, [0])
            if group_lengths is None:
                continue
            for index, response_len in enumerate(group_lengths):
                tag = {} if response_len is None else {"response_len": response_len}
                self.partitions["train"][f"{uid}_0_{index}"] = tag

    def _reward_values_by_uid(self, partition_id, uids, reward_metric=None):
        assert partition_id == "train"
        self.reward_calls.append((set(uids), reward_metric))
        return {uid: self.rewards[uid] for uid in uids if uid in self.rewards}


def _make_buffer(mixin, prompt_steps, **kwargs):
    buffer_cls = type(f"Fake{mixin.__name__}Buffer", (mixin, _FakeBufferBase), {})
    return buffer_cls(prompt_steps, **kwargs)


def _select(buffer, batch_size, sampleable_keys=None):
    if sampleable_keys is None:
        sampleable_keys = set(buffer.prompt_global_steps["train"])
    selected, _partition, _steps = buffer._select_prompt_uids("train", sampleable_keys, batch_size)
    return selected


def test_builtin_mapping_exposes_four_selection_mixins():
    assert samplers.BUILTIN_SAMPLERS == {
        "FIFO": samplers.FIFO,
        "RewardVariance": samplers.RewardVariance,
        "Mixed": samplers.Mixed,
        "Length": samplers.Length,
    }
    for mixin in samplers.BUILTIN_SAMPLERS.values():
        assert "_select_prompt_uids" in mixin.__dict__


def test_fifo_orders_by_step_then_uid_and_returns_base_snapshots():
    buffer = _make_buffer(samplers.FIFO, {"b": 2, "a": 2, "c": 1})

    selected, partition, prompt_steps = buffer._select_prompt_uids("train", {"a", "b", "c"}, 2)

    assert selected == ["c", "a"]
    assert partition == buffer.partitions["train"]
    assert partition is not buffer.partitions["train"]
    assert prompt_steps == buffer.prompt_global_steps["train"]
    assert prompt_steps is not buffer.prompt_global_steps["train"]


def test_reward_variance_orders_by_std_then_fifo_and_uid():
    buffer = _make_buffer(
        samplers.RewardVariance,
        {"high": 3, "mid": 2, "b": 1, "a": 1, "new": 4},
        rewards={
            "high": [0.0, 4.0],
            "mid": [0.0, 2.0],
            "a": [3.0, 3.0],
            "b": [3.0, 3.0],
            "new": [3.0, 3.0],
        },
    )

    assert _select(buffer, 5) == ["high", "mid", "a", "b", "new"]


def test_reward_variance_age_bonus_prevents_starvation():
    buffer = _make_buffer(
        samplers.RewardVariance,
        {"old": 0, "new": 10},
        rewards={"old": [1.0, 1.0], "new": [0.0, 2.0]},
        sampler_kwargs={"age_bonus": 0.2},
    )

    # old scores 0 + 0.2 * 10 = 2, while new scores 1 + 0.2 * 0 = 1.
    assert _select(buffer, 2) == ["old", "new"]


def test_reward_variance_cache_is_independent_incremental_and_pruned():
    buffer = _make_buffer(
        samplers.RewardVariance,
        {"a": 1, "b": 2},
        rewards={"a": [0.0, 2.0], "b": [0.0, 4.0]},
        sampler_kwargs={"reward_metric": "acc"},
    )
    buffer._dapo_classification_cache["train"] = {"a": 100.0, "b": None}

    assert _select(buffer, 1) == ["b"]
    assert buffer.reward_calls == [({"a", "b"}, "acc")]

    # A changed backing value must not invalidate an immutable finished group's cached variance.
    buffer.rewards["a"] = [0.0, 100.0]
    assert _select(buffer, 1) == ["b"]
    assert len(buffer.reward_calls) == 1

    buffer.prompt_global_steps["train"]["c"] = 3
    buffer.partitions["train"]["c_0_0"] = {"response_len": 0}
    buffer.rewards["c"] = [0.0, 6.0]
    assert _select(buffer, 3) == ["c", "b", "a"]
    assert buffer.reward_calls[-1] == ({"c"}, "acc")

    del buffer.prompt_global_steps["train"]["b"]
    del buffer.partitions["train"]["b_0_0"]
    assert _select(buffer, 2, {"a", "c"}) == ["c", "a"]
    assert set(buffer._reward_variance_cache["train"]) == {"a", "c"}
    assert buffer._dapo_classification_cache["train"] == {"a": 100.0, "b": None}


@pytest.mark.parametrize(
    ("mixin", "kwargs"),
    [(samplers.RewardVariance, {}), (samplers.Mixed, {"fifo_ratio": 0.0})],
)
@pytest.mark.parametrize("missing_values", ["absent", "empty"])
def test_reward_samplers_fail_fast_on_missing_rewards(mixin, kwargs, missing_values):
    rewards = {"a": [0.0, 1.0]}
    if missing_values == "empty":
        rewards["b"] = []
    buffer = _make_buffer(mixin, {"a": 1, "b": 2}, rewards=rewards, sampler_kwargs=kwargs)

    with pytest.raises(RuntimeError, match="missing.*values.*b"):
        _select(buffer, 1)


def test_mixed_takes_fifo_prefix_and_fills_by_priority_without_duplicates():
    buffer = _make_buffer(
        samplers.Mixed,
        {"a": 0, "b": 1, "c": 2, "d": 3},
        rewards={"a": [0.0, 0.0], "b": [0.0, 10.0], "c": [0.0, 8.0], "d": [0.0, 6.0]},
        sampler_kwargs={"fifo_ratio": 0.5},
    )

    selected = _select(buffer, 4)

    assert selected == ["a", "b", "c", "d"]
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_mixed_accepts_fifo_ratio_boundaries(ratio):
    buffer = _make_buffer(
        samplers.Mixed,
        {"a": 1},
        rewards={"a": [0.0, 1.0]},
        sampler_kwargs={"fifo_ratio": ratio},
    )
    assert _select(buffer, 1) == ["a"]


@pytest.mark.parametrize("ratio", [-0.01, 1.01, "0.5", None, True])
def test_mixed_rejects_invalid_fifo_ratio(ratio):
    buffer = _make_buffer(samplers.Mixed, {"a": 1}, sampler_kwargs={"fifo_ratio": ratio})
    with pytest.raises(ValueError, match="fifo_ratio"):
        _select(buffer, 1)


@pytest.mark.parametrize("age_bonus", [-0.1, "1", None, True, float("nan")])
def test_reward_variance_rejects_invalid_age_bonus(age_bonus):
    buffer = _make_buffer(samplers.RewardVariance, {"a": 1}, sampler_kwargs={"age_bonus": age_bonus})
    with pytest.raises(ValueError, match="age_bonus"):
        _select(buffer, 1)


@pytest.mark.parametrize(
    ("short_first", "expected"),
    [(True, ["single", "tie", "multi"]), (False, ["multi", "single", "tie"])],
)
def test_length_uses_response_len_sum_and_fifo_ties(short_first, expected):
    buffer = _make_buffer(
        samplers.Length,
        {"multi": 1, "single": 0, "tie": 2},
        lengths={"multi": [4, 4], "single": [6], "tie": [2, 4]},
        sampler_kwargs={"short_first": short_first},
    )

    # multi sums to 8 (not max=4); single and tie both sum to 6, so their steps break the tie.
    assert _select(buffer, 3) == expected


@pytest.mark.parametrize("group_lengths", [[None], None])
def test_length_fails_fast_on_missing_response_len(group_lengths):
    buffer = _make_buffer(
        samplers.Length,
        {"a": 1, "b": 2},
        lengths={"a": group_lengths, "b": [1]},
    )
    with pytest.raises(RuntimeError, match="response_len"):
        _select(buffer, 1)


@pytest.mark.parametrize("short_first", ["yes", 1, 0, None])
def test_length_rejects_non_bool_short_first(short_first):
    buffer = _make_buffer(samplers.Length, {"a": 1}, sampler_kwargs={"short_first": short_first})
    with pytest.raises(ValueError, match="short_first"):
        _select(buffer, 1)
