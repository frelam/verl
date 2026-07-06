# Copyright 2025 Individual Contributor
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
"""Trajectory collector for reject sampling.

Saves trajectories that pass the judge threshold to a JSONL file.
Each record contains the full trajectory (messages + tools) ready for
SFT data conversion.

The file path is controlled by the ``TRAJECTORY_FILE`` environment variable
(default: ``~/data/reject_sampling/collected_trajectories.jsonl``).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Optional

_DEFAULT_PATH = os.path.expanduser(
    os.environ.get("TRAJECTORY_FILE", "~/data/reject_sampling/collected_trajectories.jsonl")
)

# Thread-safe file writing
_lock = threading.Lock()


def save_trajectory(
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    data_source: str,
    score: float,
    judge_source: str,
    prompt_hash: str,
    extra: dict | None = None,
    file_path: str | None = None,
) -> None:
    """Append a trajectory to the JSONL file.

    Args:
        messages: Full message list (system, user, assistant, tool, ...).
        tools: Tool schemas used during rollout (may be empty).
        data_source: Dataset name (toolmind, terminaltraj, etc.).
        score: Judge score (0.0 - 1.0).
        judge_source: "rule" or "deepseek".
        prompt_hash: Hash of the prompt for deduplication.
        extra: Additional metadata.
        file_path: Override default file path.
    """
    path = os.path.expanduser(file_path or _DEFAULT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_source": data_source,
        "score": float(score),
        "judge_source": judge_source,
        "prompt_hash": prompt_hash,
        "messages": messages,
        "tools": tools or [],
        "extra": extra or {},
    }

    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_trajectories(file_path: str | None = None) -> list[dict[str, Any]]:
    """Read all trajectories from a JSONL file."""
    path = os.path.expanduser(file_path or _DEFAULT_PATH)
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def hash_prompt(prompt_messages: list[dict[str, Any]]) -> str:
    """Compute a stable hash of the prompt for deduplication."""
    import hashlib

    # Use only the user message content for hashing
    user_content = ""
    for msg in prompt_messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    return hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]


def get_stats(file_path: str | None = None) -> dict[str, Any]:
    """Get statistics about collected trajectories."""
    records = read_trajectories(file_path)
    if not records:
        return {"total": 0}

    from collections import Counter

    sources = Counter(r["data_source"] for r in records)
    judges = Counter(r["judge_source"] for r in records)
    scores = [r["score"] for r in records]
    unique_prompts = len(set(r["prompt_hash"] for r in records))

    return {
        "total": len(records),
        "unique_prompts": unique_prompts,
        "by_source": dict(sources),
        "by_judge": dict(judges),
        "score_mean": sum(scores) / len(scores) if scores else 0.0,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
    }
