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
"""
Preprocess logic/puzzle-domain training data for Qwen3-4B reasoning RL (DESIGN.md section 2.3).

Sources (v1 mix, DESIGN.md section 5):
  - MiniMaxAI/SynLogic (configs easy+hard, official train split) ~33k
  - BytedTsinghua-SIA/Enigmata-Data (strict train-eval separation) ~14.4k
  - Reasoning Gym procedural generation (optional, only if the `reasoning_gym`
    pip package is installed) — unbounded, capped by --rgym_per_task.

PuzzleClone is NOT in the v1 mix (DESIGN.md section 5 lists only the three
above) and ARC-AGI style tasks are already covered by both SynLogic and
Enigmata, so no standalone ARC loader is needed for v1.

Seed hard-split registry (DESIGN.md section 2.3) applies to procedurally
generated data only:
  - train seed in [0, 1e6), eval seed in [1e6, 1.1e6)
  - eval (task, config, seed) triples are written to <local_save_dir>/eval_seeds.json
  - every reasoning-gym row carries extra_info.seed; rows without a seed are
    refused (hard filter, not probabilistic exclusion)

ground_truth format (consumed by reward/compute_score.py):
  JSON string of {"answer": <str|number|nested list>, "task": <task name>}

Usage:
    python scripts/to_parquet_logic.py \
        --local_save_dir ~/data/reasoning_rl/logic
"""

import argparse
import json
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import exact_dedup, minhash_near_dedup

SYNLOGIC_REPO = "MiniMaxAI/SynLogic"
ENIGMATA_REPO = "BytedTsinghua-SIA/Enigmata-Data"

ANSWER_INSTRUCTION = (
    "Solve the problem step by step. Enclose your final answer within <answer> </answer> tags, "
    "i.e., <answer> answer here </answer>."
)

# Seed hard-split ranges (DESIGN.md section 2.3).
TRAIN_SEED_END = 1_000_000
EVAL_SEED_END = 1_100_000

# Reasoning Gym tasks used for v1 (kept small and unambiguous; extend freely).
RGYM_TASKS = [
    "sudoku",
    "mini_sudoku",
    "maze",
    "shortest_path",
    "countdown",
    "word_ladder",
    "zebra_puzzle",
    "arc_1d",
]


def make_extra_info(split, index, task_id, source, difficulty="", seed=-1):
    # Keep keys/types identical across ALL domains so per-domain parquets can be
    # concatenated later by mix.py without schema conflicts.
    return {
        "split": split,
        "index": index,
        "task_id": task_id,
        "domain": "logic",
        "source": source,
        "difficulty": str(difficulty),
        "prior_solve_rate": -1.0,
        "seed": int(seed),
    }


def _serialize_answer(answer) -> str:
    """Normalise any answer payload to a plain string for ground_truth."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, int | float | bool):
        return str(answer)
    return json.dumps(answer, ensure_ascii=False)


def make_logic_row(
    question_content: str,
    answer,
    task: str,
    task_id: str,
    source: str,
    data_source: str,
    difficulty="",
    seed=-1,
    dedup_text: str | None = None,
) -> dict | None:
    answer_str = _serialize_answer(answer)
    question_content = (question_content or "").strip()
    if not question_content or not answer_str:
        return None
    ground_truth = json.dumps({"answer": answer_str, "task": task}, ensure_ascii=False)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question_content}],
        "ability": "logic",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": make_extra_info("train", 0, task_id=task_id, source=source, difficulty=difficulty, seed=seed),
        "_dedup_text": dedup_text if dedup_text is not None else question_content,
    }


def load_synlogic(path: str | None, configs=("easy", "hard")) -> list[dict]:
    """SynLogic: prompt is already chat-format with an <answer>-tag template;
    the verifiable answer lives in extra_info.game_data_str (JSON)."""
    rows, skipped = [], 0
    for cfg in configs:
        ds = datasets.load_dataset(path or SYNLOGIC_REPO, cfg, split="train")
        for idx, ex in enumerate(ds):
            try:
                game_data = json.loads((ex.get("extra_info") or {}).get("game_data_str") or "{}")
            except (json.JSONDecodeError, TypeError):
                game_data = {}
            answer = game_data.get("answer")
            prompt = ex.get("prompt") or []
            content = prompt[0].get("content") if prompt and isinstance(prompt[0], dict) else ""
            task = str(ex.get("data_source") or "synlogic")
            row = make_logic_row(
                content,
                answer,
                task=task,
                task_id=f"synlogic-{cfg}-{(ex.get('extra_info') or {}).get('index', idx)}",
                source=f"synlogic_{cfg}",
                data_source="logic_synlogic",
                difficulty=game_data.get("difficulty", ""),
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        print(f"[synlogic/{cfg}] cumulative rows={len(rows)}, skipped so far={skipped}")
    print(f"[synlogic] loaded {len(rows)} rows, skipped {skipped} (no answer / no prompt)")
    return rows


def load_enigmata(path: str | None, max_per_task: int | None = None) -> list[dict]:
    """Enigmata-Data: raw-string prompt + exact-match answer; wrap into chat and
    append the <answer>-tag instruction (matches the official eval protocol).

    The repo stores one ``<task>/en/train.jsonl`` per task and the global
    schema cast fails on heterogeneous fields, so each task file is loaded
    separately (per-file schemas are homogeneous).
    """
    from huggingface_hub import list_repo_files

    rows, skipped = [], 0
    if path:
        task_files = [path]
    else:
        task_files = [f for f in list_repo_files(ENIGMATA_REPO, repo_type="dataset") if f.endswith("/en/train.jsonl")]
        task_files.sort()
    for rel in task_files:
        ds = (
            datasets.load_dataset(ENIGMATA_REPO, data_files=rel, split="train")
            if not path
            else datasets.load_dataset("json", data_files=rel, split="train")
        )
        n_before = len(rows)
        for idx, ex in enumerate(ds):
            if max_per_task is not None and len(rows) - n_before >= max_per_task:
                break
            question = (ex.get("prompt") or "").strip()
            answer = ex.get("answer")
            task = str(ex.get("task_name") or rel.split("/")[0])
            meta = ex.get("meta")
            meta_id = idx
            if isinstance(meta, str) and meta.strip():
                try:
                    meta_id = json.loads(meta).get("id", idx)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(meta, dict):
                meta_id = meta.get("id", idx)
            content = question + "\n\n" + ANSWER_INSTRUCTION if question else ""
            row = make_logic_row(
                content,
                answer,
                task=task,
                task_id=f"enigmata-{task}-{meta_id}",
                source="enigmata",
                data_source="logic_enigmata",
                dedup_text=question,
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        print(f"[enigmata/{rel.split('/')[0]}] +{len(rows) - n_before} rows")
    print(f"[enigmata] loaded {len(rows)} rows, skipped {skipped} (no answer / no prompt)")
    return rows


def load_reasoning_gym(per_task: int, seed_start: int, eval_registry_path: str | None) -> list[dict]:
    """Procedurally generated tasks with a seed hard-split (DESIGN.md section 2.3).

    Train seeds are drawn from [seed_start, seed_start + per_task) which must
    stay below TRAIN_SEED_END. The eval seed range [1e6, 1.1e6) is registered
    to eval_seeds.json and any generated row colliding with it is hard-filtered.
    """
    try:
        import reasoning_gym
    except ImportError:
        print("[reasoning_gym] package not installed; skipping (pip install reasoning-gym to enable)")
        return []
    if seed_start + per_task > TRAIN_SEED_END:
        raise ValueError(
            f"train seed range [{seed_start}, {seed_start + per_task}) overlaps the eval range "
            f"[{TRAIN_SEED_END}, {EVAL_SEED_END}); lower --rgym_per_task or --rgym_seed_start."
        )

    # Register the eval seed space so downstream eval generation can never
    # collide with train, and the trainer-side filter has an explicit manifest.
    eval_registry = [
        {"task": t, "config": "default", "seed": s}
        for t in RGYM_TASKS
        for s in range(TRAIN_SEED_END, min(TRAIN_SEED_END + per_task, EVAL_SEED_END))
    ]
    if eval_registry_path:
        os.makedirs(os.path.dirname(eval_registry_path), exist_ok=True)
        with open(eval_registry_path, "w") as f:
            json.dump(eval_registry, f, indent=2)
        print(f"[reasoning_gym] wrote {len(eval_registry)} eval seed triples -> {eval_registry_path}")
    eval_seeds = {(r["task"], r["seed"]) for r in eval_registry}

    rows, skipped = [], 0
    for task in RGYM_TASKS:
        try:
            data = reasoning_gym.create_dataset(task, size=per_task, seed=seed_start)
        except Exception as e:  # task renamed/removed in the installed version
            print(f"[reasoning_gym] task {task!r} unavailable: {e}; skipped")
            continue
        for i, entry in enumerate(data):
            seed = seed_start + i
            if (task, seed) in eval_seeds:  # hard filter, should never trigger
                skipped += 1
                continue
            question = (entry.get("question") or "").strip()
            answer = entry.get("answer")
            row = make_logic_row(
                question + "\n\n" + ANSWER_INSTRUCTION if question else "",
                answer,
                task=task,
                task_id=f"rgym-{task}-{seed}",
                source="reasoning_gym",
                data_source="logic_reasoning_gym",
                seed=seed,
                dedup_text=question,
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        print(f"[reasoning_gym/{task}] cumulative rows={len(rows)}")
    print(f"[reasoning_gym] loaded {len(rows)} rows, skipped {skipped}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/reasoning_rl/logic")
    parser.add_argument("--synlogic_path", default=None, help="Local path override for SynLogic.")
    parser.add_argument("--enigmata_path", default=None, help="Local path override for one Enigmata task jsonl.")
    parser.add_argument(
        "--enigmata_max_per_task",
        type=int,
        default=None,
        help="Cap rows per Enigmata task (36 tasks; debug/budget knob).",
    )
    parser.add_argument(
        "--rgym_per_task", type=int, default=1000, help="Reasoning Gym samples per task (0 disables reasoning-gym)."
    )
    parser.add_argument("--rgym_seed_start", type=int, default=0)
    parser.add_argument("--no_synlogic", action="store_true")
    parser.add_argument("--no_enigmata", action="store_true")
    parser.add_argument("--minhash_threshold", type=float, default=0.6)
    parser.add_argument("--no_minhash", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: cap rows per source run.")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    rows = []
    if not args.no_synlogic:
        rows += load_synlogic(args.synlogic_path)
    if not args.no_enigmata:
        rows += load_enigmata(args.enigmata_path, max_per_task=args.enigmata_max_per_task)
    if args.rgym_per_task > 0:
        rows += load_reasoning_gym(
            per_task=args.rgym_per_task,
            seed_start=args.rgym_seed_start,
            eval_registry_path=os.path.join(local_save_dir, "eval_seeds.json"),
        )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    # Pool-level dedup: exact text + MinHash near-dedup (procedural generators
    # can emit identical puzzles across tasks/seeds).
    texts = [r["_dedup_text"] for r in rows]
    keep = exact_dedup(texts)
    rows = [rows[i] for i in keep]
    print(f"[dedup] text exact: kept {len(rows)} / {len(texts)}")

    if not args.no_minhash:
        texts = [r["_dedup_text"] for r in rows]
        keep = minhash_near_dedup(texts, threshold=args.minhash_threshold)
        rows = [rows[i] for i in keep]
        print(f"[dedup] minhash(thr={args.minhash_threshold}): kept {len(rows)}")

    for r in rows:
        r.pop("_dedup_text")
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i

    out_ds = datasets.Dataset.from_list(rows)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "train_logic.parquet")
    out_ds.to_parquet(out_path)

    from collections import Counter

    per_source = Counter(r["extra_info"]["source"] for r in rows)
    print(f"[done] wrote {len(rows)} rows {dict(per_source)} -> {out_path}")
