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
"""Stage-4 ideal-length statistics (HALLUCINATION_RL_DESIGN.md D33).

Stage 4 starts from the DPO-trimmed checkpoint, whose main asset is a *concise*
chain of thought.  This script measures that habit on the stage-4 training mix:
every prompt is generated once by the DPO checkpoint served through vLLM, and
the response character length is aggregated per subtask (``data_source``).

The aggregated means become the per-subtask length targets of the reward's
over-length penalty (``reward/hallucination_compute_score.py``).  Outputs (all
under ``--output_dir``)::

    length_samples.jsonl   one line per generated response (lengths only)
    done.txt               task keys already generated (resume skips them)
    length_stats.json      full statistics: per data_source / domain / overall
    length_targets.json    reward-ready target file (``unit: chars``)
    length_targets.env     shell exports consumed by the stage-4 launcher

Why characters, not tokens
--------------------------
The reward sees the decoded response string and cannot cheaply re-tokenise every
rollout, so both sides must agree on one measure.  Characters are deterministic,
tokenizer-free, and already the length measure of the DPO pipeline
(``build_dataset_b.py``'s compression ratio), so the target file is generated in
characters and the reward reads ``len(solution_str)``.

Usage::

    # 1. serve the DPO checkpoint (reasoning parser REQUIRED; otherwise the
    #    think block is not separated and the response shape differs)
    vllm serve /path/to/dpo_ckpt --served-model-name dpo \\
        --reasoning-parser qwen3 --max-model-len 32768

    # 2. measure (one sample per prompt; --limit_per_source for a quick pass)
    python examples/reasoning_rl/scripts/distill/measure_length_targets.py \\
        --input ~/data/reasoning_rl/final_halluc_stage4/train.parquet \\
        --output_dir ~/data/reasoning_rl/length_targets/stage4 \\
        --base_url http://127.0.0.1:8000/v1 --model dpo \\
        --n 1 --concurrency 64

    # 3. feed the targets into stage-4 training
    source ~/data/reasoning_rl/length_targets/stage4/length_targets.env
    RESUME_PATH=<dpo_ckpt_dir> bash run_qwen3_4b_hallucination_stage4_fully_async.sh

The run is resumable: a crash only loses the in-flight batch, and re-running the
same command (or ``--aggregate_only`` after a completed run) rebuilds the
artifacts from ``length_samples.jsonl``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import random
import shlex
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import (
    append_done,
    append_jsonl,
    chat_completion,
    inject_system_prompt,
    normalise_messages,
    read_done,
    read_jsonl,
    read_parquet_rows,
    resolve_model_id,
    response_from_choice,
    summarise,
    task_key,
)

logger = logging.getLogger("measure_length_targets")

#: Target defaults, kept equal to the reward's ``_DEFAULT_LENGTH_PENALTY_*`` so a
#: generated ``length_targets.env`` fully describes the run.
DEFAULT_COEF = 0.2
DEFAULT_MAX_PENALTY = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="~/data/reasoning_rl/final_halluc_stage4/train.parquet",
        help="stage-4 parquet file / glob / directory",
    )
    parser.add_argument("--output_dir", default="~/data/reasoning_rl/length_targets/stage4")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="", help="served model id (empty = first id from GET /models)")
    parser.add_argument("--n", type=int, default=1, help="samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1, help="-1 disables top-k (vLLM extension)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="must cover the stage-4 response budget")
    parser.add_argument(
        "--enable_thinking",
        choices=("1", "0", "default"),
        default="default",
        help="chat_template_kwargs.enable_thinking. 'default' sends nothing and relies on the "
        "served template (Qwen3 defaults to thinking ON)",
    )
    parser.add_argument("--concurrency", type=int, default=64, help="in-flight generation requests")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout (s)")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--system_prompt",
        default=os.environ.get("REASONING_RL_SYSTEM_PROMPT", ""),
        help="system message prepended to every prompt; only needed if the RL run set "
        "REASONING_RL_SYSTEM_PROMPT (default: that env var, empty = no injection)",
    )
    parser.add_argument(
        "--limit_per_source",
        type=int,
        default=0,
        help="random cap of prompts per data_source (0 = every row); a quick pass over a "
        "44k-row mix should use e.g. 200",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_samples",
        type=int,
        default=20,
        help="minimum responses before a data_source/domain target is trusted; shorter "
        "sources fall back to their domain and then to the global mean",
    )
    parser.add_argument(
        "--target_metric",
        choices=("mean", "p50", "p90"),
        default="mean",
        help="statistic written as the ideal length (default: mean, as specified for D33)",
    )
    parser.add_argument(
        "--keep_truncated",
        action="store_true",
        help="count responses that hit --max_tokens (default: drop them; their true length "
        "is unknown and they would inflate the target)",
    )
    parser.add_argument("--coef", type=float, default=DEFAULT_COEF, help="exported LENGTH_PENALTY_COEF")
    parser.add_argument("--max_penalty", type=float, default=DEFAULT_MAX_PENALTY, help="exported LENGTH_PENALTY_MAX")
    parser.add_argument("--log_every", type=int, default=50, help="progress every N prompts")
    parser.add_argument("--no_resume", action="store_true", help="ignore done markers and regenerate everything")
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="skip generation and rebuild the stats/targets from length_samples.jsonl",
    )
    return parser


def _domain_of(row: dict) -> str:
    info = row.get("extra_info") or {}
    return str(row.get("ability") or info.get("domain") or "")


def _select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Apply the optional per-source cap with a seeded, reproducible draw."""
    if args.limit_per_source <= 0:
        return rows
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[str(row.get("data_source") or "")].append(index)
    rng = random.Random(args.seed)
    chosen: list[int] = []
    for indices in buckets.values():
        take = min(args.limit_per_source, len(indices))
        chosen.extend(rng.sample(indices, take))
    chosen.sort()
    logger.info(
        "per-source cap %d: %d -> %d rows (%d sources)",
        args.limit_per_source,
        len(rows),
        len(chosen),
        len(buckets),
    )
    return [rows[index] for index in chosen]


def _generate_one(row: dict, args: argparse.Namespace, model_id: str) -> tuple[dict, list[dict], str | None]:
    """One prompt -> N responses, reduced to their lengths (no text is stored)."""
    messages = inject_system_prompt(normalise_messages(row.get("prompt")), args.system_prompt)
    if not messages or not any(message.get("content", "").strip() for message in messages):
        return row, [], "empty prompt"

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
        return row, [], str(exc)

    choices = result.get("choices") or []
    completion_tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
    samples = []
    for choice in choices:
        response = response_from_choice(choice)
        if not response.strip():
            continue
        finish_reason = str(choice.get("finish_reason") or "")
        samples.append(
            {
                "response_chars": len(response),
                # ``usage`` is per request, not per choice: divide as an estimate
                # (tokens are reported for context only; the target is characters).
                "response_tokens_est": completion_tokens // max(1, len(choices)) or max(1, len(response) // 4),
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
            }
        )
    return row, samples, None


def _pick(stats: dict, metric: str) -> float:
    return float(stats["mean"] if metric == "mean" else stats[metric])


def _aggregate(samples: list[dict], args: argparse.Namespace) -> tuple[dict, dict]:
    """Raw statistics + the reward-ready target payload."""
    by_source: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    source_domain: dict[str, str] = {}
    dropped_truncated = 0
    for sample in samples:
        if not args.keep_truncated and sample.get("truncated"):
            dropped_truncated += 1
            continue
        chars = int(sample.get("response_chars") or 0)
        if chars <= 0:
            continue
        source = str(sample.get("data_source") or "__unknown__")
        domain = str(sample.get("domain") or "__unknown__")
        by_source[source].append(chars)
        by_domain[domain].append(chars)
        source_domain.setdefault(source, domain)

    overall = summarise(value for values in by_source.values() for value in values)
    if overall["count"] == 0:
        raise SystemExit("no usable responses to aggregate (everything truncated or empty?)")

    default_target = int(round(_pick(overall, args.target_metric)))
    domain_stats = {domain: summarise(values) for domain, values in sorted(by_domain.items())}
    domain_targets = {
        domain: int(round(_pick(stats, args.target_metric)))
        for domain, stats in domain_stats.items()
        if stats["count"] >= args.min_samples
    }

    source_stats = {}
    targets = {}
    for source, values in sorted(by_source.items()):
        stats = summarise(values)
        domain = source_domain.get(source, "")
        if stats["count"] >= args.min_samples:
            target, target_from = int(round(_pick(stats, args.target_metric))), "source"
        elif domain in domain_targets:
            target, target_from = domain_targets[domain], "domain"
        else:
            target, target_from = default_target, "default"
        targets[source] = target
        source_stats[source] = {**stats, "target": target, "target_from": target_from, "domain": domain}

    targets["__default__"] = default_target
    stats_payload = {
        "config": {
            "input": args.input,
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "keep_truncated": bool(args.keep_truncated),
            "target_metric": args.target_metric,
            "min_samples": args.min_samples,
            "unit": "chars",
        },
        "counts": {
            "responses_used": overall["count"],
            "truncated_dropped": dropped_truncated,
            "sources": len(by_source),
            "domains": len(by_domain),
        },
        "overall": overall,
        "by_domain": domain_stats,
        "by_source": source_stats,
    }
    targets_payload = {
        "unit": "chars",
        "metric": args.target_metric,
        "drop_truncated": not args.keep_truncated,
        "n": args.n,
        "input": args.input,
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "min_samples": args.min_samples,
        "coef": args.coef,
        "max_penalty": args.max_penalty,
        "targets": targets,
        "by_domain": domain_targets,
    }
    return stats_payload, targets_payload


def _write_artifacts(args: argparse.Namespace, samples_path: str, model_id: str) -> dict:
    samples = read_jsonl(samples_path)
    if not samples:
        raise SystemExit(f"no samples in {samples_path}; run the generation step first")
    stats_path = os.path.join(args.output_dir, "length_stats.json")
    targets_path = os.path.join(args.output_dir, "length_targets.json")
    env_path = os.path.join(args.output_dir, "length_targets.env")
    if not model_id:
        # --aggregate_only has no endpoint to ask; keep the model of the run that
        # produced the samples.
        try:
            with open(targets_path, encoding="utf-8") as handle:
                model_id = str(json.load(handle).get("model") or "")
        except (OSError, ValueError):
            model_id = ""
    stats_payload, targets_payload = _aggregate(samples, args)
    targets_payload["model"] = model_id or "unknown"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, ensure_ascii=False, indent=2)
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(targets_payload, handle, ensure_ascii=False, indent=2)
    with open(env_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage-4 ideal-length targets (HALLUCINATION_RL_DESIGN.md D33).\n")
        handle.write("# source this file, then launch the stage-4 recipe.\n")
        handle.write(f"export LENGTH_PENALTY_TARGETS_PATH={shlex.quote(os.path.abspath(targets_path))}\n")
        handle.write(f"export LENGTH_PENALTY_COEF={args.coef}\n")
        handle.write(f"export LENGTH_PENALTY_MAX={args.max_penalty}\n")

    logger.info(
        "[targets] %d sources, default=%d chars; stats -> %s, targets -> %s, env -> %s",
        len(targets_payload["targets"]) - 1,
        targets_payload["targets"]["__default__"],
        stats_path,
        targets_path,
        env_path,
    )
    for source, target in sorted(targets_payload["targets"].items()):
        if source == "__default__":
            continue
        info = stats_payload["by_source"][source]
        logger.info(
            "[targets] %-28s target=%6d chars (from %-7s, n=%d, mean=%.0f, p90=%.0f)",
            source,
            target,
            info["target_from"],
            info["count"],
            info["mean"],
            info["p90"],
        )
    return targets_payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.input = os.path.expanduser(args.input)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    samples_path = os.path.join(args.output_dir, "length_samples.jsonl")
    done_path = os.path.join(args.output_dir, "done.txt")

    if args.aggregate_only:
        _write_artifacts(args, samples_path, model_id="")
        return 0

    if args.system_prompt:
        logger.info("system prompt: %.120s", args.system_prompt)
    else:
        logger.info("system prompt: none (no injection)")

    rows = read_parquet_rows(args.input)
    logger.info("loaded %d rows from %s", len(rows), args.input)
    rows = _select_rows(rows, args)
    rows = [row for row in rows if row.get("prompt")]
    logger.info(
        "sources: %s",
        json.dumps(dict(Counter(str(row.get("data_source") or "") for row in rows)), ensure_ascii=False),
    )

    done = set() if args.no_resume else read_done(done_path)
    pending = [row for row in rows if task_key(row) not in done]
    logger.info("pending prompts: %d (%d already done)", len(pending), len(rows) - len(pending))
    if not pending:
        _write_artifacts(args, samples_path, model_id="")
        return 0

    model_id = resolve_model_id(args.base_url, args.api_key, args.model)
    logger.info("served model: %s", model_id)

    counters = Counter()
    started = time.time()
    last_logged = 0
    batch_size = max(1, args.concurrency)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            generated = list(pool.map(lambda row: _generate_one(row, args, model_id), batch))

        finished_rows = []
        finished_done = []
        for row, samples, error in generated:
            key = task_key(row)
            if error:
                counters["generation_errors"] += 1
                logger.warning("[gen] %s: %s", key, error)
                # Not marked done: an HTTP failure is retried on the next run.
                continue
            for index, sample in enumerate(samples):
                finished_rows.append(
                    {
                        "task_id": key,
                        "data_source": str(row.get("data_source") or ""),
                        "domain": _domain_of(row),
                        "gen_index": index,
                        **sample,
                    }
                )
            counters["prompts_done"] += 1
            counters["responses"] += len(samples)
            counters["truncated"] += sum(1 for sample in samples if sample["truncated"])
            finished_done.append(key)

        append_jsonl(samples_path, finished_rows)
        append_done(done_path, finished_done)

        if counters["prompts_done"] - last_logged >= max(1, args.log_every):
            last_logged = counters["prompts_done"]
            elapsed = time.time() - started
            rate = counters["prompts_done"] / max(elapsed, 1e-9)
            logger.info(
                "[progress] prompts=%d/%d responses=%d truncated=%d rate=%.2f prompt/s eta=%.1f min errors=%d",
                counters["prompts_done"],
                len(pending),
                counters["responses"],
                counters["truncated"],
                rate,
                (len(pending) - counters["prompts_done"]) / max(rate, 1e-9) / 60.0,
                counters["generation_errors"],
            )

    run_stats = dict(counters)
    run_stats["elapsed_minutes"] = (time.time() - started) / 60.0
    with open(os.path.join(args.output_dir, "run_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(run_stats, handle, ensure_ascii=False, indent=2)
    logger.info("run stats: %s", json.dumps(run_stats, ensure_ascii=False))

    targets_payload = _write_artifacts(args, samples_path, model_id)
    logger.info(
        "next: source %s, then RESUME_PATH=<dpo_ckpt_dir> bash run_qwen3_4b_hallucination_stage4_fully_async.sh",
        os.path.join(args.output_dir, "length_targets.env"),
    )
    logger.info("targets written: %s", json.dumps(targets_payload["targets"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
