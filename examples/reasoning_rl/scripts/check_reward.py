#!/usr/bin/env python3
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
"""Self-check for a reasoning_rl parquet (train or val) before blaming the model.

A rule-verifiable pool is only as good as the pair *(row, verifier)*.  When one
``data_source`` scores 0 while the others train, the cause is usually one of:

1. the stored ground truth cannot be produced from the prompt (wrong column,
   empty answer, mis-serialised payload);
2. the reward function cannot *recognise* a correct answer (missing marker,
   non-unique answer compared literally, seed/metadata lost).

Both are visible without a GPU: feed each row's own ground truth back as the
model answer ("echo") and require ``score == 1.0``.  This script prints the
per-source composition and the echo pass rate, so a broken source is obvious
before spending a training run on it::

    # a mixed val/train file, all sources
    python3 examples/reasoning_rl/scripts/check_reward.py --data_file $HOME/data/reasoning_rl/final_v2/val.parquet

    # one suspicious source, more samples in the report
    python3 examples/reasoning_rl/scripts/check_reward.py \
        --data_file $HOME/data/reasoning_rl/final_v2/val.parquet \
        --data_source logic_reasoning_gym --samples 5

    # real generations dumped by trainer.validation_data_dir=/tmp/val_dump
    python3 examples/reasoning_rl/scripts/check_reward.py --dump_dir /tmp/val_dump

Echo is defined per domain: ``<answer>{gt}</answer>`` *and* the bare ``{gt}``
for ``logic_*`` (both are legitimate response shapes and both must score 1.0),
``\\boxed{gt}`` for ``math_*``/``stem_*``.  ``code_*`` (needs a sandbox) and
``if_*`` (the constraint JSON is not an answer) are counted but not echoed.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reward"))
# The reward dispatcher imports ``verl.utils.reward_score`` for math/stem, so the
# checkout root must be importable even when the script runs from elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mix import read_parquet_rows  # noqa: E402

ECHOABLE_PREFIXES = ("math", "stem", "logic")
SKIPPED_PREFIXES = ("code", "if")


def ground_truth_answer(ground_truth) -> str | None:
    """Model-visible answer inside a ground_truth payload (``None`` if absent)."""
    payload = ground_truth
    if isinstance(ground_truth, str):
        try:
            payload = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return ground_truth.strip() or None
    if isinstance(payload, dict):
        payload = payload.get("answer")
    if payload is None:
        return None
    text = payload.strip() if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return text or None


def echo_response(data_source: str, ground_truth, style: str = "tagged") -> str | None:
    """A compliant response that repeats the ground truth, or ``None`` if the
    domain cannot be echoed offline (code needs the sandbox, if needs a real
    constraint-satisfying answer).

    ``style`` mirrors the two shapes a logic answer legitimately takes: the
    reasoning_rl ``<answer>`` contract (``tagged``) and the bare answer several
    task prompts explicitly demand instead (``bare``).  Both must score 1.0;
    a source that only passes one of them is a reward-side bug, not a model
    problem.
    """
    answer = ground_truth_answer(ground_truth)
    if answer is None:
        return None
    prefix = data_source.split("_", 1)[0]
    if prefix in SKIPPED_PREFIXES:
        return None
    if prefix == "logic":
        open_tag = "" if style == "bare" else "<answer>"
        close_tag = "" if style == "bare" else "</answer>"
        return f"<think>echo</think>\n\n{open_tag}{answer}{close_tag}"
    return f"<think>echo</think>\n\nThe final answer is: \\boxed{{{answer}}}"


def audit_rows(rows: list[dict], compute_score, samples: int = 3) -> dict:
    """Echo every row through the real reward function and collect a report."""
    report: dict = {"counts": collections.Counter(), "sources": collections.defaultdict(collections.Counter)}
    per_source: dict[str, dict] = collections.defaultdict(
        lambda: {
            "echoed": 0,
            "passed": 0,
            "bare_echoed": 0,
            "bare_passed": 0,
            "empty_gt": 0,
            "skipped": 0,
            "failures": [],
            "tasks": collections.Counter(),
        }
    )

    for row in rows:
        data_source = row.get("data_source", "unknown")
        ground_truth = (row.get("reward_model") or {}).get("ground_truth")
        entry = per_source[data_source]
        report["counts"][data_source] += 1
        report["sources"][data_source][(row.get("extra_info") or {}).get("source", "unknown")] += 1

        if isinstance(ground_truth, str):
            try:
                payload = json.loads(ground_truth)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("task"):
                entry["tasks"][str(payload["task"])] += 1

        if data_source.startswith(SKIPPED_PREFIXES):
            entry["skipped"] += 1  # code (sandbox) / if (constraint payload, not an answer)
            continue
        if ground_truth_answer(ground_truth) is None:
            entry["empty_gt"] += 1
            continue
        styles = (("tagged", ""), ("bare", "bare_")) if data_source.startswith("logic") else (("tagged", ""),)
        for style, key in styles:
            response = echo_response(data_source, ground_truth, style)
            if response is None:
                entry["skipped"] += 1
                continue
            entry[f"{key}echoed"] += 1
            score = compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=row.get("extra_info") or {},
            )
            score = score.get("score", 0.0) if isinstance(score, dict) else score
            if score == 1.0:
                entry[f"{key}passed"] += 1
            elif len(entry["failures"]) < samples:
                entry["failures"].append(
                    {
                        "style": style,
                        "task": (json.loads(ground_truth).get("task") if _is_json_object(ground_truth) else None),
                        "ground_truth": str(ground_truth)[:200],
                        "response": response[-200:],
                        "score": score,
                    }
                )

    report["per_source"] = dict(per_source)
    return report


def _is_json_object(text) -> bool:
    if not isinstance(text, str):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def print_report(report: dict, samples: int = 3) -> None:
    print(f"[check_reward] {sum(report['counts'].values())} rows")
    for data_source, count in sorted(report["counts"].items()):
        entry = report["per_source"][data_source]
        sources = ", ".join(f"{k}={v}" for k, v in sorted(report["sources"][data_source].items()))
        words = []
        if entry["empty_gt"]:
            words.append(f"EMPTY GROUND TRUTH x{entry['empty_gt']}")
        for key, label in (("", "echo"), ("bare_", "bare echo")):
            if not entry[f"{key}echoed"]:
                continue
            rate = entry[f"{key}passed"] / entry[f"{key}echoed"]
            status = "OK" if entry[f"{key}passed"] == entry[f"{key}echoed"] else "FAIL"
            words.append(f"{label} {entry[f'{key}passed']}/{entry[f'{key}echoed']} ({rate:.0%}) {status}")
        if entry["skipped"]:
            words.append(f"echo skipped x{entry['skipped']} (needs sandbox/constraints)")
        if not entry["echoed"] and not entry["skipped"]:
            words.append("nothing echoed")
        if entry["tasks"]:
            words.append("tasks: " + ", ".join(f"{k}={v}" for k, v in sorted(entry["tasks"].items())))
        print(f"  {data_source}: {count} rows [{sources}] -> {' | '.join(words)}")
        for failure in entry["failures"][:samples]:
            print(f"      score={failure['score']} style={failure['style']} task={failure['task']}")
            print(f"        gt  : {failure['ground_truth']!r}")
            print(f"        echo: {failure['response']!r}")


def audit_dump(dump_dir: str, task_filter: str | None = None) -> None:
    """Summarise ``trainer.validation_data_dir`` JSONL dumps (real generations)."""
    files = sorted(glob.glob(os.path.join(dump_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"[check_reward] no *.jsonl found under {dump_dir}")
    stats: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "reward": 0.0, "zero_samples": []})
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ground_truth = item.get("gts")
                task = None
                if _is_json_object(ground_truth):
                    task = json.loads(ground_truth).get("task")
                key = str(task) if task else "unparsed-ground-truth"
                if task_filter and task_filter not in key:
                    continue
                stat = stats[key]
                stat["n"] += 1
                stat["reward"] += float(item.get("score", 0.0))
                if float(item.get("score", 0.0)) == 0.0 and len(stat["zero_samples"]) < 2:
                    stat["zero_samples"].append(item.get("output", ""))
    print(f"[check_reward] {len(files)} dump file(s) under {dump_dir}")
    for key, stat in sorted(stats.items()):
        print(f"  {key}: n={stat['n']} mean_reward={stat['reward'] / stat['n']:.3f}")
        for output in stat["zero_samples"]:
            print(f"      zero-scored output: {str(output)[-300:]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_file", default=None, help="Parquet file / glob / directory to audit.")
    parser.add_argument("--data_source", default=None, help="Only audit this data_source (exact match).")
    parser.add_argument("--samples", type=int, default=3, help="Failure samples printed per data source.")
    parser.add_argument("--dump_dir", default=None, help="Also summarise dumped val generations (*.jsonl).")
    parser.add_argument("--dump_task", default=None, help="Filter the dump summary by ground-truth task substring.")
    args = parser.parse_args(argv)

    if not args.data_file and not args.dump_dir:
        parser.error("pass --data_file and/or --dump_dir")

    if args.data_file:
        from compute_score import compute_score

        rows = read_parquet_rows(args.data_file)
        if args.data_source:
            rows = [r for r in rows if r.get("data_source") == args.data_source]
            if not rows:
                raise SystemExit(f"[check_reward] no rows with data_source={args.data_source!r}")
        print_report(audit_rows(rows, compute_score, samples=args.samples), samples=args.samples)

    if args.dump_dir:
        audit_dump(args.dump_dir, task_filter=args.dump_task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
