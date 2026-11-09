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
"""Selection-only mixins for the built-in V1 replay-buffer samplers."""

from numbers import Real
from statistics import pstdev


def _selection_snapshots(buffer, partition_id: str) -> tuple[dict[str, dict], dict[str, int]]:
    return dict(buffer.partitions[partition_id]), dict(buffer.prompt_global_steps[partition_id])


def _fifo_order(sampleable_keys: set[str], prompt_global_steps: dict[str, int]) -> list[str]:
    return sorted(sampleable_keys, key=lambda uid: (prompt_global_steps.get(uid, 0), uid))


def _numeric_sampler_kwarg(sampler_kwargs, name: str, default: float) -> float:
    value = sampler_kwargs.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    return float(value)


def _fifo_ratio(sampler_kwargs) -> float:
    ratio = _numeric_sampler_kwarg(sampler_kwargs, "fifo_ratio", 0.5)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"fifo_ratio must be in [0, 1], got {ratio}")
    return ratio


def _age_bonus(sampler_kwargs) -> float:
    age_bonus = _numeric_sampler_kwarg(sampler_kwargs, "age_bonus", 0.0)
    if age_bonus < 0 or age_bonus != age_bonus:
        raise ValueError(f"age_bonus must be non-negative, got {age_bonus}")
    return age_bonus


class FIFO:
    """Select the oldest prompt groups, with uid providing deterministic ties."""

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        partition_snapshot, prompt_global_steps_snapshot = _selection_snapshots(self, partition_id)
        ordered_keys = _fifo_order(sampleable_keys, prompt_global_steps_snapshot)
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot


class RewardVariance:
    """Prioritize reward variance while giving older prompt groups an age bonus."""

    def _reward_variances(self, partition_id: str, uids: set[str]) -> dict[str, float]:
        caches = getattr(self, "_reward_variance_cache", None)
        if caches is None:
            caches = {}
            self._reward_variance_cache = caches
        cache = caches.setdefault(partition_id, {})

        # Finished groups are immutable. Drop entries only after their group leaves the active metadata.
        active_uids = set(self.prompt_global_steps[partition_id])
        active_uids.update(key.split("_")[0] for key in self.partitions[partition_id])
        for uid in cache.keys() - active_uids:
            del cache[uid]

        missing_uids = uids - cache.keys()
        if missing_uids:
            reward_metric = self.sampler_kwargs.get("reward_metric")
            values_by_uid = self._reward_values_by_uid(partition_id, missing_uids, reward_metric=reward_metric)
            missing_rewards = {uid for uid in missing_uids if not values_by_uid.get(uid)}
            if missing_rewards:
                metric_name = reward_metric or "reward"
                raise RuntimeError(
                    f"Reward sampler is missing {metric_name!r} values for prompt groups: {sorted(missing_rewards)}"
                )
            for uid in missing_uids:
                cache[uid] = float(pstdev(values_by_uid[uid]))

        return {uid: cache[uid] for uid in uids}

    def _reward_variance_order(
        self, partition_id: str, sampleable_keys: set[str], prompt_global_steps: dict[str, int]
    ) -> list[str]:
        age_bonus = _age_bonus(self.sampler_kwargs)
        variances = self._reward_variances(partition_id, sampleable_keys)
        # A common current-step offset does not affect ordering. The newest known dispatch step therefore
        # supplies the reference without extending the selection-only base-class interface.
        current_global_step = max(prompt_global_steps.values(), default=0)
        scores = {
            uid: variances[uid] + age_bonus * (current_global_step - prompt_global_steps.get(uid, 0))
            for uid in sampleable_keys
        }
        return sorted(
            sampleable_keys,
            key=lambda uid: (-scores[uid], prompt_global_steps.get(uid, 0), uid),
        )

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        partition_snapshot, prompt_global_steps_snapshot = _selection_snapshots(self, partition_id)
        ordered_keys = self._reward_variance_order(partition_id, sampleable_keys, prompt_global_steps_snapshot)
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot


class Mixed(RewardVariance):
    """Take a FIFO prefix, then fill the batch by reward-variance priority without duplicates."""

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        fifo_ratio = _fifo_ratio(self.sampler_kwargs)
        partition_snapshot, prompt_global_steps_snapshot = _selection_snapshots(self, partition_id)
        fifo_order = _fifo_order(sampleable_keys, prompt_global_steps_snapshot)
        fifo_count = int(fifo_ratio * batch_size)
        selected = fifo_order[:fifo_count]
        selected_set = set(selected)

        priority_order = self._reward_variance_order(partition_id, sampleable_keys, prompt_global_steps_snapshot)
        for uid in priority_order:
            if len(selected) >= batch_size:
                break
            if uid not in selected_set:
                selected.append(uid)
                selected_set.add(uid)

        return selected[:batch_size], partition_snapshot, prompt_global_steps_snapshot


class Length:
    """Order prompt groups by the sum of their trajectories' response lengths."""

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        short_first = self.sampler_kwargs.get("short_first", True)
        if not isinstance(short_first, bool):
            raise ValueError(f"short_first must be a bool, got {short_first!r}")

        partition_snapshot, prompt_global_steps_snapshot = _selection_snapshots(self, partition_id)
        response_lengths: dict[str, int] = {}
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid not in sampleable_keys:
                continue
            if "response_len" not in tag:
                raise RuntimeError(f"Length sampler requires trajectory tag 'response_len', missing on {key!r}")
            response_lengths[uid] = response_lengths.get(uid, 0) + tag["response_len"]

        missing_uids = sampleable_keys - response_lengths.keys()
        if missing_uids:
            raise RuntimeError(
                f"Length sampler found no trajectory response_len values for prompt groups: {sorted(missing_uids)}"
            )

        direction = 1 if short_first else -1
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda uid: (
                direction * response_lengths[uid],
                prompt_global_steps_snapshot.get(uid, 0),
                uid,
            ),
        )
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot


BUILTIN_SAMPLERS = {
    "FIFO": FIFO,
    "RewardVariance": RewardVariance,
    "Mixed": Mixed,
    "Length": Length,
}

__all__ = ["BUILTIN_SAMPLERS", "FIFO", "Length", "Mixed", "RewardVariance"]
