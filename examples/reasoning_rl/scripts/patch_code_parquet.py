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
"""Apply the post-audit code-data fixes to an already-built code parquet.

Why
---
``to_parquet_code.py`` rebuilds from the raw sources, but two of those loaders
are broken with current tooling: ``codeparrot/apps`` is a script dataset
(``datasets>=4`` refuses it) and ``agentica-org/DeepCoder-Preview-Dataset`` now
requires an explicit config.  When you already have the previous
``train_code.parquet``, its rows are exactly the ones you trained on, so the two
data-side fixes can be applied in place instead of re-downloading anything:

1. **name the function in the prompt** -- ``fn_name`` was already stored in
   ``reward_model.ground_truth``; the old prompt only said "Implement the
   required function(s)", so the model could not know the exact name the
   verifier looks up.
2. **unquote string returns** -- the old canonicaliser stored a list-wrapped
   string return as ``json.dumps("hi") == '"hi"'``, but sandbox_fusion prints
   ``str(result) == "hi"`` and compares stdout as text, so every test failed.

The rows, order and ``extra_info`` are preserved.  Sharding is preserved too:
a multi-shard input produces one output file per input file, which keeps each
Arrow string column under the 2 GB offset limit and lets ``mix_replay.py`` read
the output directory directly.

Caveat for (2)
--------------
From the processed parquet alone it is not always possible to tell a
list-wrapped string return ``["hi"]`` (should become ``hi``) from a raw string
return whose *content* is the two characters ``"hi"`` (should stay ``"hi"``):
both were stored byte-identically.  The script unquotes the JSON-string form,
which is correct for the list-wrapped convention that dominates; a string return
whose value is itself a JSON-string literal is vanishingly rare and would lose
its outer quotes.  Rebuilding from raw avoids the ambiguity entirely.

Usage::

    # single file
    python3 examples/reasoning_rl/scripts/patch_code_parquet.py \
        --input  ~/data/reasoning_rl/code/train_code.parquet \
        --output ~/data/reasoning_rl/code_v2/train_code.parquet

    # sharded input (00000.parquet, 00001.parquet, ...): pass the directory,
    # a glob, or several files; --output may also be a directory
    python3 examples/reasoning_rl/scripts/patch_code_parquet.py \
        --input  ~/data/reasoning_rl/code \
        --output ~/data/reasoning_rl/code_v2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from to_parquet_code import FUNCTION_INSTRUCTION  # noqa: E402  (new template, single source of truth)

# The instruction string written by every pre-fix line of to_parquet_code.py.
OLD_FUNCTION_INSTRUCTION = "Implement the required function(s). Wrap your code in ```python and ```."


def _unquote_string_return(value):
    """``'"hi"'`` -> ``'hi'``; every other canonical form is left alone.

    ``json.loads`` returns a ``str`` only for the JSON-string form, i.e. exactly
    the old ``json.dumps`` of a list-wrapped string return.  int/list/bool/None
    forms load to a non-str and plain unquoted strings fail to parse, so both are
    preserved.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    return parsed if isinstance(parsed, str) else value


def patch_row(row: dict) -> tuple[bool, int]:
    """Apply both fixes to one row in place; return ``(prompt_changed, n_unquoted)``."""
    raw = row.get("reward_model", {}).get("ground_truth")
    try:
        ground_truth = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return False, 0
    if not isinstance(ground_truth, dict) or not ground_truth.get("fn_name"):
        return False, 0

    fn_name = ground_truth["fn_name"]
    new_instruction = FUNCTION_INSTRUCTION.replace("{fn_name}", str(fn_name))

    prompt_changed = False
    prompt = row.get("prompt")
    if (
        isinstance(prompt, list)
        and prompt
        and isinstance(prompt[0], dict)
        and isinstance(prompt[0].get("content"), str)
    ):
        content = prompt[0]["content"]
        if new_instruction not in content:
            if OLD_FUNCTION_INSTRUCTION in content:
                content = content.replace(OLD_FUNCTION_INSTRUCTION, new_instruction)
            else:
                content = content.rstrip() + "\n\n" + new_instruction
            prompt[0]["content"] = content
            prompt_changed = True

    outputs = ground_truth.get("outputs")
    n_unquoted = 0
    if isinstance(outputs, list):
        fixed = []
        for value in outputs:
            new_value = _unquote_string_return(value)
            if new_value != value:
                n_unquoted += 1
            fixed.append(new_value)
        if n_unquoted:
            ground_truth["outputs"] = fixed
            row["reward_model"]["ground_truth"] = json.dumps(ground_truth)

    return prompt_changed, n_unquoted


def _collect_input_files(inputs) -> list[str]:
    """Expand files/directories/globs into a de-duplicated, ordered file list.

    Accepts a single path or a list of them.  A directory is scanned recursively
    for ``*.parquet`` (the ``00000.parquet``, ``00001.parquet``, ... shard
    layout), sorted by name so the row order is deterministic.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    files: list[str] = []
    for raw in inputs:
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
            if not found:
                raise SystemExit(f"[patch_code] no *.parquet found under {path}")
            files.extend(found)
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise SystemExit(f"[patch_code] input not found: {path}")

    seen: set[str] = set()
    unique: list[str] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def patch_dataset(inputs, output_path: str, dry_run: bool = False) -> dict:
    """Patch every input shard, writing one output file per input shard.

    Rows are processed shard by shard on purpose.  Concatenating every shard into
    one ``Dataset.from_list`` builds a single Arrow string array whose 32-bit
    offsets overflow once the combined text passes ~2 GB
    (``pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays``).
    Keeping the input sharding avoids that and keeps memory bounded.
    """
    import datasets

    files = _collect_input_files(inputs)
    output_arg = os.path.abspath(os.path.expanduser(output_path))
    output_is_file = output_arg.endswith(".parquet")
    if len(files) > 1 and output_is_file:
        raise SystemExit(
            "[patch_code] multiple input shards with a single .parquet --output would concatenate every row "
            "into one Arrow string column and can hit pyarrow's offset overflow; pass a directory to --output "
            "so the input sharding is preserved."
        )
    out_dir = os.path.dirname(output_arg) if output_is_file else output_arg

    stats = {
        "input_files": files,
        "input_shards": len(files),
        "output_files": [],
        "total_rows": 0,
        "call_based_rows": 0,
        "prompts_updated": 0,
        "outputs_unquoted": 0,
        "rows_touched": 0,
    }

    used_outputs: set[str] = set()
    for index, shard in enumerate(files):
        rows = datasets.load_dataset("parquet", data_files=shard, split="train").to_list()
        for row in rows:
            changed, unquoted = patch_row(row)
            gt = row.get("reward_model", {}).get("ground_truth")
            try:
                has_fn = isinstance(gt, str) and "fn_name" in json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                has_fn = False
            stats["call_based_rows"] += int(has_fn)
            stats["prompts_updated"] += int(changed)
            stats["outputs_unquoted"] += unquoted
            stats["rows_touched"] += int(changed or unquoted > 0)
        stats["total_rows"] += len(rows)

        if len(files) == 1:
            out_file = output_arg if output_is_file else os.path.join(out_dir, "train_code.parquet")
        else:
            out_file = os.path.join(out_dir, os.path.basename(shard))
            if out_file in used_outputs:
                out_file = os.path.join(out_dir, f"{index:05d}_{os.path.basename(shard)}")
        used_outputs.add(out_file)
        stats["output_files"].append(out_file)

        if not dry_run:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            datasets.Dataset.from_list(rows).to_parquet(out_file)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more pre-fix parquet files, or directories containing *.parquet shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Patched parquet path (single input), or a directory (one output file per input shard).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Report counts without writing anything.")
    args = parser.parse_args(argv)

    stats = patch_dataset(args.input, args.output, dry_run=args.dry_run)
    if args.dry_run:
        print("[patch_code] DRY RUN (nothing written)")
    else:
        print(f"[patch_code] wrote {len(stats['output_files'])} file(s)")
    for path in stats["input_files"]:
        print(f"[patch_code] input shard:  {path}")
    for path in stats["output_files"]:
        print(f"[patch_code] output shard: {path}")
    for key in ("input_shards", "total_rows", "call_based_rows", "prompts_updated", "outputs_unquoted", "rows_touched"):
        print(f"[patch_code] {key}={stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
