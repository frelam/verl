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
"""Official-verifier bridge for the Reasoning Gym domain (``logic_reasoning_gym``).

Why this exists
---------------
``to_parquet_logic.py`` stores the generator's own ``answer`` field, which for
several Reasoning Gym tasks is only *one* of many correct answers::

    countdown       "6*(98 - 54) - 34 - 66"   arithmetic expression, many equivalents
    word_ladder     "PEGS,BEGS,BEES"          any valid ladder is correct
    shortest_path   "left up up up"           any equally short path is correct

The generic ``logic_answer_match`` string comparison therefore scores a correct
response 0 whenever the model writes an equivalent-but-different answer.  The
library ships a per-task ``score_answer(answer, entry)`` (the verifier its own
evals use) which understands the task semantics; DESIGN.md section 6 specifies
``logic_reasoning_gym -> reasoning_gym 库的 verify()``, and this module is that
hook.

How the entry is recovered (no data rebuild needed)
---------------------------------------------------
Reasoning Gym datasets derive item ``idx`` from ``Random(self.seed + idx)``, and
``to_parquet_logic.py`` stores the item seed in ``extra_info.seed``.  A cached
per-thread dataset created with ``create_dataset(task, size=1, seed=0)``
therefore regenerates the exact puzzle the model was shown by setting
``dataset.seed = <row seed>`` and reading item 0.  Before scoring, the
regenerated ``answer`` must equal the stored ground truth: if the installed
``reasoning-gym`` version drifted (or the seed came from another generator run)
the entry is rejected and the caller falls back to string comparison, so a
mismatch can never inflate the reward.

Contract (mirrors ``verify_enigmata``): ``True``/``False`` when the official
verifier owns the sample, ``None`` when it does not (package missing,
regeneration failed, ground truth disagreement).  Only a full official score
(``>= 1.0``) counts as correct -- partial credit does not fit the binary
pass-rate reward contract.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# One dataset per (thread, task). ``dataset.seed`` is mutated per call, so the
# instances must never be shared across threads: the reward loop scores samples
# concurrently through a thread pool.
_LOCAL = threading.local()


def _get_dataset(task: str):
    """Per-thread Reasoning Gym dataset for ``task``, or ``None`` if unavailable."""
    try:
        import reasoning_gym
    except ImportError:  # package only needed by the data-prep side / this bridge
        logger.warning("[reasoning_rl] reasoning_gym not installed; falling back to string comparison")
        return None
    cache = getattr(_LOCAL, "datasets", None)
    if cache is None:
        cache = _LOCAL.datasets = {}
    if task not in cache:
        try:
            cache[task] = reasoning_gym.create_dataset(task, size=1, seed=0)
        except Exception as e:  # unknown/renamed task in the installed version
            logger.warning("[reasoning_rl] reasoning_gym task %r unavailable: %s", task, e)
            cache[task] = None
    return cache[task]


def _regenerated_entry(dataset, seed: int):
    """Item ``seed`` of the generator (``Random(self.seed + idx)``)."""
    dataset.seed = seed
    return dataset[0]


def _answers_agree(regenerated, stored) -> bool:
    return str(regenerated).strip() == str(stored).strip()


def score_with_dataset(dataset, prediction: str | None, answer, seed) -> bool | None:
    """Score ``prediction`` with an already-built dataset (injectable for tests).

    ``dataset`` only needs ``seed``, ``__getitem__`` and ``score_answer``.
    """
    if prediction is None or seed is None:
        return None
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return None
    if seed < 0:  # -1 is the "no seed" sentinel written by the prep script
        return None
    try:
        entry = _regenerated_entry(dataset, seed)
    except Exception as e:
        logger.warning("[reasoning_rl] reasoning_gym regeneration failed for seed %s: %s", seed, e)
        return None
    if not isinstance(entry, dict) or not _answers_agree(entry.get("answer"), answer):
        # Version drift or a seed from another generator run: refuse to score.
        logger.info("[reasoning_rl] reasoning_gym entry for seed %s does not match the stored answer", seed)
        return None
    try:
        return float(dataset.score_answer(prediction, entry)) >= 1.0
    except Exception as e:
        logger.warning("[reasoning_rl] reasoning_gym score_answer failed for seed %s: %s", seed, e)
        return None


def verify_reasoning_gym(prediction: str | None, answer, task: str | None, seed) -> bool | None:
    """Official Reasoning Gym verdict, or ``None`` when this module cannot judge."""
    if not task:
        return None
    dataset = _get_dataset(str(task))
    if dataset is None:
        return None
    return score_with_dataset(dataset, prediction, answer, seed)


def seed_from_extra_info(extra_info) -> int | None:
    """Generator seed of a reasoning-gym row.

    ``to_parquet_logic.py`` writes it to ``extra_info.seed``; the ``rgym-<task>-<seed>``
    task id is the fallback for rows that lack the field.  ``None`` when neither
    is usable, which disables official verification for that row.
    """
    if not isinstance(extra_info, dict):
        return None
    seed = extra_info.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    task_id = extra_info.get("task_id")
    if isinstance(task_id, str) and task_id.startswith("rgym-"):
        head, _, tail = task_id.rpartition("-")
        if head and tail.isdigit():
            return int(tail)
    return None
