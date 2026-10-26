#!/usr/bin/env python3
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
"""Shared helpers for the reasoning-efficiency DPO data pipeline.

Pipeline (three scripts, one purpose: build preference pairs where a *shorter,
just as correct* chain of thought is preferred over the original one)::

    collect_dataset_a.py   stage-3 rows -> vLLM (n samples) -> reward == 1 -> A
    build_dataset_b.py     A -> judge picks the best correct rollout -> judge
                              rewrites its <think> block -> reward re-check -> B
    build_dpo_pairs.py     A + B -> (prompt, chosen=B, rejected=A) -> DPO parquet

The three scripts must agree on three things, which is why they live here:

* how a verl parquet row's ``prompt`` becomes the chat messages sent to vLLM
  (``normalise_messages`` + ``inject_system_prompt``),
* how a response splits into the ``<think>`` body and the final answer
  (``split_think`` / ``join_think`` -- deliberately the same contract as
  ``compute_score.format_ok``),
* how one sample is identified and checkpointed (``task_key`` / ``make_sample_id``
  plus the JSONL + ``.done`` files, so a crashed collection resumes cleanly).

Nothing here talks to a model; the two HTTP helpers are the only network code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root, so ``examples.reasoning_rl.reward.*`` imports work no matter
# which directory the script was launched from.  ``scripts/hallucination`` hosts
# the parquet schema/writer used by the stage-3 builder; reusing it keeps the
# Arrow schema of this pipeline identical to the training artifacts.
REPO_ROOT = Path(__file__).resolve().parents[4]
for _path in (
    REPO_ROOT,
    REPO_ROOT / "examples" / "reasoning_rl" / "scripts",
    REPO_ROOT / "examples" / "reasoning_rl" / "scripts" / "hallucination",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# parquet / jsonl IO
# ---------------------------------------------------------------------------


def _resolve_parquet_files(path: str) -> list[str]:
    import glob

    expanded = os.path.expanduser(path)
    if os.path.isdir(expanded):
        files = sorted(glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True))
    else:
        files = sorted(glob.glob(expanded))
    if not files:
        raise FileNotFoundError(f"no parquet files matched: {path}")
    return files


def read_parquet_rows(path: str) -> list[dict]:
    """Read a parquet file / glob / directory into Python dicts, batch by batch.

    Same contract as ``scripts/mix.py``'s reader: ``iter_batches`` keeps Arrow's
    32-bit string offsets local, so the multi-GB stage-3 artifact (and the
    rollout dumps built from it) loads without an overflow.
    """
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for file in _resolve_parquet_files(path):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=8192):
            rows.extend(batch.to_pylist())
    return rows


def write_parquet_rows(rows: list[dict], path: str) -> None:
    """Chunked parquet write through the stage-2/3 schema writer.

    ``schema.write_rows_parquet`` infers the Arrow schema from a probe that
    covers every populated field, which is what lets prompt-bearing rows with
    heterogeneous ``extra_info`` values share one artifact.
    """
    from schema import write_rows_parquet

    write_rows_parquet(rows, os.path.expanduser(path))


def append_jsonl(path: str, rows: list[dict]) -> None:
    """Append finished rows to a JSONL checkpoint file (one flush per call)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping a torn final line left by a crash."""
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_done(path: str) -> set[str]:
    """Read the ``.done`` marker file (one task key per line)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_done(path: str, task_ids: list[str]) -> None:
    if not task_ids:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(task_ids) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# prompts and the think-block contract
# ---------------------------------------------------------------------------


def normalise_messages(prompt) -> list[dict]:
    """Verl ``prompt`` column -> OpenAI chat messages.

    The stage-1/2/3 artifacts already store a message list; a plain string is
    accepted as well so the pipeline can be pointed at a raw question dump.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages: list[dict] = []
    for message in prompt or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        messages.append(
            {
                "role": str(message.get("role") or "user"),
                "content": "" if content is None else str(content),
            }
        )
    return messages


def inject_system_prompt(messages: list[dict], system_prompt: str) -> list[dict]:
    """Prepend a system message, mirroring ``ReasoningRLDataset``.

    A prompt that already opens with a system message is left untouched (two
    system turns render unpredictably), which is also what the RL dataset does.
    """
    system_prompt = (system_prompt or "").strip()
    if not system_prompt or (messages and messages[0].get("role") == "system"):
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def prompt_question_text(messages: list[dict]) -> str:
    """The system+user part of a prompt (what the judge needs to see)."""
    parts = [m.get("content", "") for m in messages if m.get("role") in ("system", "user")]
    return "\n\n".join(part for part in parts if part.strip())


def split_think(response: str) -> tuple[str, str] | None:
    """Split a compliant response into ``(think_body, post_think_tail)``.

    Contract (identical to ``compute_score.format_ok``): the response starts
    with ``<think>``, has exactly one non-empty think body, a non-empty final
    answer, and no stray think tags afterwards.  A response that fails this is
    not usable by the pipeline -- dataset A only keeps reward==1 rows, and the
    reward's format gate already rejects these, but the check is cheap and
    makes the scripts safe to run on hand-edited data.
    """
    if not isinstance(response, str) or not response.startswith(THINK_OPEN):
        return None
    body, separator, tail = response[len(THINK_OPEN) :].partition(THINK_CLOSE)
    if not separator or not body.strip() or not tail.strip():
        return None
    if THINK_OPEN in tail or THINK_CLOSE in tail:
        return None
    return body, tail


def join_think(body: str, tail: str) -> str:
    """Stitch a rewritten think body back onto the *original* tail.

    The tail (final-answer section) is never regenerated by the judge: copying
    it byte-for-byte is what guarantees the rewritten response keeps the exact
    ``\\boxed{...}`` answer of the original.
    """
    return f"{THINK_OPEN}\n{body.strip()}\n{THINK_CLOSE}{tail}"


# ---------------------------------------------------------------------------
# row identity / sharding / resume
# ---------------------------------------------------------------------------


def task_key(row: dict) -> str:
    """Stable per-question key.

    Stage-1/2/3 rows carry ``extra_info.task_id`` (globally unique by contract).
    A hand-made row may not, so fall back to a hash of the prompt.
    """
    task_id = str((row.get("extra_info") or {}).get("task_id") or "").strip()
    if task_id:
        return task_id
    payload = json.dumps(row.get("prompt"), ensure_ascii=False, sort_keys=True)
    return "anon-" + hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def make_sample_id(task_id: str, index: int) -> str:
    """One rollout of one question; the join key of A and B."""
    return f"{task_id}#{index:02d}"


def stable_shard(task_id: str, num_shards: int) -> int:
    """Deterministic, disjoint shard assignment (md5, not Python's salted hash)."""
    if num_shards <= 1:
        return 0
    digest = hashlib.md5(task_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def ground_truth_of(row: dict) -> str:
    """The ground-truth string of a verl row (or of an A/B output row)."""
    if "ground_truth" in row and row["ground_truth"] is not None:
        value = row["ground_truth"]
    else:
        value = (row.get("reward_model") or {}).get("ground_truth")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------


def load_reward():
    """Load the stage-2/3 reward dispatcher (``{"score": float}``)."""
    from examples.reasoning_rl.reward.hallucination_compute_score import compute_score

    return compute_score


def load_verifier_helpers():
    """``format_ok`` (format gate) and ``extract_final`` (answer comparison)."""
    from examples.reasoning_rl.reward.compute_score import format_ok
    from examples.reasoning_rl.reward.hallucination_compute_score import extract_final

    return format_ok, extract_final


def score_response(
    reward_fn,
    row: dict,
    response: str,
    sandbox_fusion_url: str | None = None,
) -> float:
    """Score one response exactly as the RL trainer would, failing closed to 0."""
    kwargs = {}
    if sandbox_fusion_url:
        kwargs["sandbox_fusion_url"] = sandbox_fusion_url
    try:
        result = reward_fn(
            data_source=str(row.get("data_source") or ""),
            solution_str=response,
            ground_truth=ground_truth_of(row),
            extra_info=row.get("extra_info") or {},
            **kwargs,
        )
    except Exception:  # noqa: BLE001 - a broken row must not kill the collection
        logger.exception("[reward] scoring crashed for %s", row.get("data_source"))
        return 0.0
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP client (vLLM rollout service / LLM judge service)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _session():
    import requests

    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def chat_completion(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: float = 1800.0,
    max_retries: int = 3,
    label: str = "llm",
) -> dict:
    """POST ``/chat/completions`` with retries on transient failures.

    ``requests.Session`` is thread-local: the collection scripts hammer the
    service from a thread pool, and a shared Session is not guaranteed safe.
    """
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _session().post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            message = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code in (408, 409, 425, 429) or 500 <= response.status_code < 600:
                last_error = RuntimeError(message)
            else:
                raise RuntimeError(f"{label} request rejected ({message})")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(30.0, 2.0**attempt) + random.random())
    raise RuntimeError(f"{label} request failed after {max_retries + 1} attempts: {last_error}")


def resolve_model_id(base_url: str, api_key: str, model: str, timeout: float = 60.0) -> str:
    """Explicit ``--model`` or the first id advertised by ``GET /models``."""
    if model:
        return model
    import requests

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json().get("data") or []
    if not data:
        raise RuntimeError("service advertises no model; pass --model explicitly")
    return data[0]["id"]


def response_from_choice(choice: dict) -> str:
    """Rebuild the full response text from one chat completion choice.

    vLLM may be started with ``--reasoning-parser qwen3``, in which case the
    think block arrives as ``message.reasoning_content`` and ``content`` holds
    only the final answer.  The reward expects the template shape
    ``<think>...</think>\\n\\n{answer}``, so reconstruct it.  When the server
    returns the raw text (no parser), the content is already complete.
    """
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if content.lstrip().startswith(THINK_OPEN):
        return content.lstrip()
    if reasoning:
        body = content.lstrip()
        if body.startswith(THINK_CLOSE):
            body = body[len(THINK_CLOSE) :].lstrip("\n")
        return f"{THINK_OPEN}\n{reasoning.strip()}\n{THINK_CLOSE}\n\n{body}"
    return content


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def summarise(values) -> dict:
    """count / mean / min / p50 / p90 / max, tolerating empty input."""
    numbers = sorted(float(v) for v in values if v is not None)
    if not numbers:
        return {"count": 0}
    count = len(numbers)

    def quantile(percent: float) -> float:
        return numbers[min(count - 1, int(percent * (count - 1)))]

    return {
        "count": count,
        "mean": sum(numbers) / count,
        "min": numbers[0],
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "max": numbers[-1],
    }
