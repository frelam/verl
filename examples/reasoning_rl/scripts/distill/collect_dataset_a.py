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
"""Dataset A: correct rollouts of the RL checkpoint on the stage-3 data.

For every row of the stage-3 mix, send the (already template-rendered) prompt to
the vLLM service that serves the RL-finished checkpoint ``--n`` times, score
every returned trajectory with the same reward the RL run used, and keep the
trajectories that score 1.  Sampling N times per prompt is what fixes the "one
sample was wrong, but the prompt is still useful" case: the prompt is only
dropped when none of its N samples is correct.

Scoring is the stage-2/3 dispatcher itself
(``examples/reasoning_rl/reward/hallucination_compute_score.py``), so dataset A
is by construction "what the RL run considers correct" -- format gate,
abstention contracts and the repetition penalty included.

Output layout (all under ``--output_dir``)::

    samples.shard{idx}.jsonl   one line per kept trajectory, written as prompts finish
    done.shard{idx}.txt        task keys already processed (resume skips them)
    dataset_a.parquet          merged A artifact (all shards in the directory)
    a_stats.json               counts, correctness rate and length distributions

Usage::

    python examples/reasoning_rl/scripts/distill/collect_dataset_a.py \
        --input ~/data/reasoning_rl/final_halluc_stage3/train.parquet \
        --output_dir ~/data/reasoning_rl/dpo/stage_a \
        --base_url http://127.0.0.1:8000/v1 \
        --n 8 --concurrency 64 --score_workers 8

Multi-process collection: run the same command in several processes with a
distinct ``--shard_index`` / ``--num_shards`` pair and ``--no_convert``; the
shard assignment is an md5 of ``task_id`` so the splits are stable and disjoint.
After all shards finish, run once more with ``--convert_only`` to write
``dataset_a.parquet``.

Notes
-----
* The service must be OpenAI-compatible (vLLM's ``/v1``); ``--model`` defaults
  to the first id returned by ``GET /models``.
* ``--system_prompt`` is OPTIONAL and only needed when the RL run itself
  injected ``REASONING_RL_SYSTEM_PROMPT``.  That system message is added at
  training time and is not stored in the parquet rows, so a run that never set
  the variable (the default) needs no system prompt here either.  When given
  (argument or env var), it is prepended to every prompt unless the row already
  carries one.
* ``code_*`` rows are skipped unless ``--sandbox_fusion_url`` is given: their
  reward only works against a code sandbox, and scoring them without one would
  silently discard every correct generation.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import (
    append_done,
    append_jsonl,
    chat_completion,
    ground_truth_of,
    inject_system_prompt,
    load_reward,
    make_sample_id,
    normalise_messages,
    read_done,
    read_jsonl,
    read_parquet_rows,
    resolve_model_id,
    response_from_choice,
    score_response,
    stable_shard,
    summarise,
    task_key,
    write_parquet_rows,
)

logger = logging.getLogger("collect_dataset_a")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="~/data/reasoning_rl/final_halluc_stage3/train.parquet",
        help="stage-3 parquet file / glob / directory",
    )
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/dpo/stage_a")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="", help="served model id (empty = first id from GET /models)")
    parser.add_argument("--n", type=int, default=8, help="samples per prompt; only correct ones are kept")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1, help="-1 disables top-k (vLLM extension)")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--enable_thinking",
        choices=("1", "0", "default"),
        default="default",
        help="chat_template_kwargs.enable_thinking. 'default' sends nothing and relies on the "
        "served template (Qwen3 defaults to thinking ON); '1'/'0' forces the flag",
    )
    parser.add_argument("--concurrency", type=int, default=64, help="in-flight generation requests")
    parser.add_argument(
        "--score_workers",
        type=int,
        default=8,
        help="reward-scoring threads (sympy/sandbox are CPU-bound)",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout (s)")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--system_prompt",
        default=os.environ.get("REASONING_RL_SYSTEM_PROMPT", ""),
        help="system message prepended to every prompt; only needed if the RL run set "
        "REASONING_RL_SYSTEM_PROMPT (default: that env var, empty = no injection)",
    )
    parser.add_argument(
        "--abilities",
        default="",
        help="comma list to keep (math,logic,stem,code,commonsense,if); empty = all",
    )
    parser.add_argument(
        "--sandbox_fusion_url",
        default="",
        help="code reward sandbox; code rows are skipped when empty",
    )
    parser.add_argument("--min_score", type=float, default=1.0, help="keep trajectories with score >= this")
    parser.add_argument("--max_per_prompt", type=int, default=0, help="cap kept trajectories per prompt (0 = keep all)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many prompts per shard (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20, help="progress every N prompts")
    parser.add_argument("--no_resume", action="store_true", help="ignore done markers and redo everything")
    parser.add_argument(
        "--no_convert",
        action="store_true",
        help="skip the final parquet conversion (parallel shards)",
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="only merge the JSONL shards into dataset_a.parquet",
    )
    return parser


def _generate_one(row: dict, args: argparse.Namespace, model_id: str) -> dict:
    """One prompt -> N sampled trajectories (no scoring yet)."""
    messages = inject_system_prompt(normalise_messages(row.get("prompt")), args.system_prompt)
    if not messages or not any(message.get("content", "").strip() for message in messages):
        return {"row": row, "messages": messages, "samples": [], "error": "empty prompt"}

    payload = {
        "model": model_id,
        "messages": messages,
        "n": max(1, args.n),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.top_k is not None and args.top_k >= 0:
        payload["top_k"] = args.top_k
    if args.enable_thinking != "default":
        payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking == "1"}

    try:
        result = chat_completion(
            args.base_url,
            args.api_key,
            payload,
            timeout=args.timeout,
            max_retries=args.max_retries,
            label="vllm",
        )
    except Exception as exc:  # noqa: BLE001 - one bad prompt must not stop the run
        return {"row": row, "messages": messages, "samples": [], "error": str(exc)}

    usage = result.get("usage") or {}
    samples = []
    for choice in result.get("choices") or []:
        response = response_from_choice(choice)
        if not response.strip():
            continue
        samples.append(
            {
                "response": response,
                "finish_reason": str(choice.get("finish_reason") or ""),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                # ``usage`` is per request; per-choice counts are not reported, so
                # this is an estimate used only for ordering/statistics.
                "response_tokens": int(usage.get("completion_tokens") or 0) // max(1, len(result.get("choices") or [1]))
                or max(1, len(response) // 4),
            }
        )
    return {"row": row, "messages": messages, "samples": samples, "error": None}


def _build_stats(rows: list[dict]) -> dict:
    by_source = Counter(row.get("data_source", "") for row in rows)
    by_ability = Counter(row.get("ability", "") for row in rows)
    tasks = {row.get("task_id") for row in rows}
    scores = Counter(f"{float(row.get('score', 0.0)):.2f}" for row in rows)
    return {
        "kept_trajectories": len(rows),
        "prompts": len(tasks),
        "trajectories_per_prompt": summarise(
            Counter(row.get("task_id") for row in rows).values()
        ),
        "by_data_source": dict(by_source),
        "by_ability": dict(by_ability),
        "kept_score": dict(scores),
        "response_tokens": summarise(row.get("response_tokens", 0) for row in rows),
        "response_chars": summarise(len(row.get("response", "")) for row in rows),
    }


def _convert(output_dir: str) -> list[dict]:
    """Merge every ``samples*.jsonl`` shard in ``output_dir`` into one artifact."""
    rows: list[dict] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(output_dir, "samples*.jsonl"))):
        shard_rows = read_jsonl(path)
        duplicate = 0
        for row in shard_rows:
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in seen:
                duplicate += 1
                continue
            seen.add(sample_id)
            rows.append(row)
        logger.info("[convert] %s: %d rows (%d duplicate sample_ids skipped)", path, len(shard_rows), duplicate)
    if not rows:
        raise SystemExit(f"no samples found under {output_dir}; nothing to convert")
    rows.sort(key=lambda row: (row.get("task_id", ""), row.get("sample_id", "")))
    write_parquet_rows(rows, os.path.join(output_dir, "dataset_a.parquet"))
    stats = _build_stats(rows)
    with open(os.path.join(output_dir, "a_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    logger.info("[convert] wrote %d rows to %s/dataset_a.parquet", len(rows), output_dir)
    logger.info("[convert] stats: %s", json.dumps(stats, ensure_ascii=False))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.input = os.path.expanduser(args.input)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.convert_only:
        _convert(args.output_dir)
        return 0

    if args.system_prompt:
        logger.info("system prompt: %.120s", args.system_prompt)
    else:
        logger.info("system prompt: none (no injection)")

    rows = read_parquet_rows(args.input)
    logger.info("loaded %d rows from %s", len(rows), args.input)

    if args.abilities:
        allowed = {item.strip() for item in args.abilities.split(",") if item.strip()}
        before = len(rows)
        rows = [row for row in rows if str(row.get("ability") or "") in allowed]
        logger.info("ability filter %s: %d -> %d rows", sorted(allowed), before, len(rows))

    if not args.sandbox_fusion_url:
        code_rows = [row for row in rows if str(row.get("data_source") or "").startswith("code")]
        if code_rows:
            logger.warning(
                "skipping %d code rows: their reward needs --sandbox_fusion_url",
                len(code_rows),
            )
            rows = [row for row in rows if not str(row.get("data_source") or "").startswith("code")]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.num_shards > 1:
        rows = [row for row in rows if stable_shard(task_key(row), args.num_shards) == args.shard_index]
        logger.info("shard %d/%d: %d rows", args.shard_index, args.num_shards, len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]

    samples_path = os.path.join(args.output_dir, f"samples.shard{args.shard_index}.jsonl")
    done_path = os.path.join(args.output_dir, f"done.shard{args.shard_index}.txt")
    done = set() if args.no_resume else read_done(done_path)
    pending = [row for row in rows if task_key(row) not in done]
    logger.info("pending prompts: %d (%d already done)", len(pending), len(rows) - len(pending))
    if not pending:
        if not args.no_convert:
            _convert(args.output_dir)
        return 0

    model_id = resolve_model_id(args.base_url, args.api_key, args.model)
    logger.info("served model: %s", model_id)
    reward_fn = load_reward()

    counters = Counter()
    started = time.time()
    last_logged = 0
    batch_size = max(1, args.concurrency)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            generated = list(pool.map(lambda row: _generate_one(row, args, model_id), batch))

        # (row, message-list, sample, original index) in a stable order.
        scoring_tasks = []
        for item in generated:
            row = item["row"]
            if item["error"]:
                counters["generation_errors"] += 1
                logger.warning("[gen] %s: %s", task_key(row), item["error"])
                continue
            for index, sample in enumerate(item["samples"]):
                scoring_tasks.append((row, item["messages"], sample, index))

        with ThreadPoolExecutor(max_workers=max(1, args.score_workers)) as pool:
            scores = list(
                pool.map(
                    lambda task: score_response(
                        reward_fn, task[0], task[2]["response"], args.sandbox_fusion_url or None
                    ),
                    scoring_tasks,
                )
            )

        # Group the scored trajectories back per prompt, keep only correct ones.
        per_prompt: dict[str, list[tuple[int, dict, float]]] = {}
        prompt_rows: dict[str, dict] = {}
        messages_by_prompt: dict[str, list[dict]] = {}
        for (row, messages, sample, index), score in zip(scoring_tasks, scores, strict=True):
            key = task_key(row)
            per_prompt.setdefault(key, []).append((index, sample, score))
            prompt_rows[key] = row
            messages_by_prompt[key] = messages
            counters["generated"] += 1

        finished_rows = []
        finished_done = []
        for key in per_prompt:
            row = prompt_rows[key]
            kept = [(index, sample, score) for index, sample, score in per_prompt[key] if score >= args.min_score]
            kept.sort(key=lambda item: (item[1]["response_tokens"], len(item[1]["response"])))
            if args.max_per_prompt > 0:
                kept = kept[: args.max_per_prompt]
            summary = dict(
                data_source=str(row.get("data_source") or ""),
                ability=str(row.get("ability") or ""),
                ground_truth=ground_truth_of(row),
                extra_info=row.get("extra_info") or {},
            )
            for index, sample, score in kept:
                finished_rows.append(
                    {
                        "sample_id": make_sample_id(key, index),
                        "task_id": key,
                        "gen_index": index,
                        "prompt": messages_by_prompt[key],
                        "response": sample["response"],
                        "score": float(score),
                        "prompt_tokens": sample["prompt_tokens"],
                        "response_tokens": sample["response_tokens"],
                        "finish_reason": sample["finish_reason"],
                        "gen_model": model_id,
                        **summary,
                    }
                )
            counters["kept"] += len(kept)
            counters["prompts_done"] += 1
            if not kept:
                counters["prompts_without_correct_sample"] += 1
            finished_done.append(key)

        # Rows are only persisted once the whole prompt is finished, so a crash
        # can never leave a half-written prompt behind that resume would skip.
        append_jsonl(samples_path, finished_rows)
        append_done(done_path, finished_done)

        if counters["prompts_done"] - last_logged >= max(1, args.log_every):
            last_logged = counters["prompts_done"]
            elapsed = time.time() - started
            rate = counters["prompts_done"] / max(elapsed, 1e-9)
            logger.info(
                "[progress] prompts=%d/%d kept=%d generated=%d rate=%.2f prompt/s eta=%.1f min errors=%d",
                counters["prompts_done"],
                len(pending),
                counters["kept"],
                counters["generated"],
                rate,
                (len(pending) - counters["prompts_done"]) / max(rate, 1e-9) / 60.0,
                counters["generation_errors"],
            )

    run_stats = dict(counters)
    run_stats["correct_rate"] = counters["kept"] / max(1, counters["generated"])
    run_stats["elapsed_minutes"] = (time.time() - started) / 60.0
    stats_path = os.path.join(args.output_dir, f"a_run_stats.shard{args.shard_index}.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(run_stats, handle, ensure_ascii=False, indent=2)
    logger.info("run stats: %s", json.dumps(run_stats, ensure_ascii=False))

    if not args.no_convert:
        _convert(args.output_dir)
    else:
        logger.info("--no_convert: run this script again with --convert_only after every shard finishes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
