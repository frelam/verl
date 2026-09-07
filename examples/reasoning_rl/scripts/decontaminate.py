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
Two-stage train-vs-validation decontamination pipeline (DESIGN.md section 3), fully automatic.

Stage A: word 8-gram overlap — catches verbatim/near-verbatim leaked problems.
Stage B: embedding cosine via a locally served embedding model — catches paraphrased
         contamination (the dominant form for web-distilled corpora like Dr.SCI).

No manual spot-checking: the cosine threshold is auto-calibrated per run with
Youden's J over a positive-control distribution (perturbed copies of validation
problems = "must be deleted") and a negative-control distribution (random
train x val pairs). Deletions are written to an audit CSV for post-hoc review.

Prerequisite: serve the embedding model locally, e.g.
    vllm serve Qwen/Qwen3-Embedding-8B --task embed --port 8001

Usage:
    python scripts/decontaminate.py \
        --train_parquet ~/data/reasoning_rl/math/train_math.parquet \
        --val_file ~/data/reasoning_rl/val/aime_2024_2025.parquet \
        --val_file ~/data/reasoning_rl/val/math_test.parquet \
        --out_dir ~/data/reasoning_rl/math/decontaminated
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time

import datasets
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import normalize_text

NGRAM_N = 8
VAL_TEXT_CANDIDATES = ["problem", "question", "Problem", "Question", "content"]

# Instruction suffixes appended by the to_parquet_* scripts; stripped before
# similarity computation so they don't inflate every cosine equally.
INSTRUCTION_SUFFIXES = [
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Write a complete Python program that reads the input from standard input",
    "Implement the required function(s).",
]

_DIGIT_MAP = str.maketrans("0123456789", "1234567890")
_PUNCT_RE = re.compile(r"[^\w\s]")


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def strip_instruction(text: str) -> str:
    for suffix in INSTRUCTION_SUFFIXES:
        idx = text.find(suffix)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def word_ngrams(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def perturb(text: str, rng: random.Random) -> str:
    """Positive-control generator: meaning-preserving surface edits that mimic
    paraphrase-level contamination (digit swap, punctuation/case noise, word drop)."""
    t = text.translate(_DIGIT_MAP)
    if rng.random() < 0.5:
        t = _PUNCT_RE.sub(lambda m: m.group(0) if rng.random() < 0.5 else " ", t)
    if rng.random() < 0.5:
        t = t.lower() if rng.random() < 0.5 else t
    words = t.split()
    if len(words) > 10 and rng.random() < 0.5:
        del words[rng.randrange(len(words))]
    return " ".join(words)


# --------------------------------------------------------------------------- #
# embedding client (OpenAI-compatible /v1/embeddings served by vLLM)
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self, base_url: str, model: str, batch_size: int = 64, max_retries: int = 3):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim(texts)), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            out[start : start + len(batch)] = self._post(batch)
            if (start // self.batch_size) % 20 == 0:
                print(f"  embedded {start + len(batch)} / {len(texts)}", flush=True)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def _dim(self, texts: list[str]) -> int:
        return self._post(texts[:1]).shape[1]

    def _post(self, batch: list[str]) -> np.ndarray:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url,
                    json={"model": self.model, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                return np.asarray([d["embedding"] for d in data], dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"embedding request failed after {self.max_retries} retries: {last_err}")


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_val_problems(files: list[str], text_field: str | None) -> list[str]:
    problems = []
    for f in files:
        if f.endswith(".parquet"):
            ds = datasets.load_dataset("parquet", data_files=f, split="train")
        else:
            ds = datasets.load_dataset("json", data_files=f, split="train")
        field = text_field
        if field is None:
            for cand in VAL_TEXT_CANDIDATES:
                if cand in ds.column_names:
                    field = cand
                    break
        if field is None:
            raise ValueError(f"{f}: cannot find a text field among {VAL_TEXT_CANDIDATES}; pass --val_text_field")
        problems.extend(strip_instruction(str(v)) for v in ds[field])
        print(f"[val] loaded {len(ds)} problems from {f} (field={field})")
    return problems


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--val_file", action="append", required=True, help="Repeatable. parquet/json/jsonl.")
    parser.add_argument("--val_text_field", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--embedding_base_url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--threshold", default="auto", help="'auto' or a fixed float like 0.85")
    parser.add_argument("--ngram_n", type=int, default=NGRAM_N)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = datasets.load_dataset("parquet", data_files=args.train_parquet, split="train")
    train_texts = [strip_instruction(p[0]["content"]) for p in train_ds["prompt"]]
    val_problems = load_val_problems(args.val_file, args.val_text_field)
    print(f"[load] train={len(train_texts)} val={len(val_problems)}")

    dropped: dict[int, dict] = {}  # train idx -> audit record

    # ---------------- Stage A: n-gram overlap ---------------- #
    val_ngrams: set[str] = set()
    for vp in val_problems:
        val_ngrams.update(word_ngrams(vp, args.ngram_n))
    print(f"[ngram] val {args.ngram_n}-gram set size: {len(val_ngrams)}")

    for i, tt in enumerate(train_texts):
        if word_ngrams(tt, args.ngram_n) & val_ngrams:
            dropped[i] = {"stage": "ngram", "cosine": "", "matched_val_index": -1}
    print(f"[ngram] flagged {len(dropped)} train rows")

    # ---------------- Stage B: embedding cosine ---------------- #
    embedder = Embedder(args.embedding_base_url, args.embedding_model, batch_size=args.batch_size)

    print("[embed] encoding validation set + positive controls ...")
    positives = [perturb(vp, rng) for vp in val_problems]
    val_emb = embedder.encode(val_problems)  # (V, D) normalized
    pos_emb = embedder.encode(positives)

    # Stream the train pool: encode one chunk at a time and immediately reduce
    # against val_emb, so we never materialize the full (N, D) train matrix
    # (N=460k x D=4096 fp32 would be ~7.5GB). Chunk size adapts to |val| so the
    # (chunk, V) sims matrix stays under ~256MB.
    n_val = len(val_problems)
    chunk = max(256, min(8192, int(256_000_000 / (4 * n_val))))
    top1 = np.empty(len(train_texts), dtype=np.float32)
    top1_idx = np.empty(len(train_texts), dtype=np.int32)
    print(f"[embed] streaming train pool (chunk={chunk}) ...")
    for s in range(0, len(train_texts), chunk):
        emb = embedder.encode(train_texts[s : s + chunk])  # (c, D) normalized
        sims = emb @ val_emb.T  # (c, V)
        top1[s : s + chunk] = sims.max(axis=1)
        top1_idx[s : s + chunk] = sims.argmax(axis=1)

    # ---------------- threshold calibration ---------------- #
    if args.threshold == "auto":
        pos_cos = np.sum(pos_emb * val_emb, axis=1)  # cosine(orig, perturbed copy)
        n_neg = min(2000, len(train_texts))
        neg_rows = rng.sample(range(len(train_texts)), n_neg)
        neg_emb = embedder.encode([train_texts[i] for i in neg_rows])  # small, fits memory
        neg_cos = (neg_emb @ val_emb.T).max(axis=1)  # random-pair upper tail
        best_t, best_j = 0.85, -1.0
        for t in np.arange(0.70, 0.951, 0.01):
            tpr = float(np.mean(pos_cos >= t))
            fpr = float(np.mean(neg_cos >= t))
            if tpr - fpr > best_j:
                best_j, best_t = tpr - fpr, float(t)
        threshold = min(max(best_t, 0.80), 0.90)  # sanity band
        print(
            f"[calib] pos mean={pos_cos.mean():.3f} neg p99={np.percentile(neg_cos, 99):.3f} "
            f"-> threshold={threshold:.2f} (J={best_j:.3f})"
        )
    else:
        threshold = float(args.threshold)
        print(f"[calib] fixed threshold={threshold}")

    for i in range(len(train_texts)):
        if i not in dropped and top1[i] >= threshold:
            dropped[i] = {"stage": "embedding", "cosine": f"{top1[i]:.4f}", "matched_val_index": int(top1_idx[i])}

    # ---------------- outputs ---------------- #
    keep_idx = [i for i in range(len(train_texts)) if i not in dropped]
    name = os.path.splitext(os.path.basename(args.train_parquet))[0]

    out_parquet = os.path.join(args.out_dir, f"{name}.decontaminated.parquet")
    train_ds.select(keep_idx).to_parquet(out_parquet)

    audit_csv = os.path.join(args.out_dir, f"{name}.dropped.csv")
    with open(audit_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_index", "task_id", "stage", "cosine", "matched_val_index", "train_text_head"])
        for i in sorted(dropped):
            rec = dropped[i]
            writer.writerow(
                [
                    i,
                    train_ds[i]["extra_info"].get("task_id", ""),
                    rec["stage"],
                    rec["cosine"],
                    rec["matched_val_index"],
                    train_texts[i][:200],
                ]
            )

    report = {
        "train_parquet": args.train_parquet,
        "val_files": args.val_file,
        "n_train": len(train_texts),
        "n_val": len(val_problems),
        "threshold": threshold,
        "dropped_ngram": sum(1 for r in dropped.values() if r["stage"] == "ngram"),
        "dropped_embedding": sum(1 for r in dropped.values() if r["stage"] == "embedding"),
        "kept": len(keep_idx),
    }
    with open(os.path.join(args.out_dir, f"{name}.report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[done] {json.dumps(report)}")
    print(f"[done] kept parquet -> {out_parquet}\n[done] audit csv -> {audit_csv}")
