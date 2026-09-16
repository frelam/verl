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

The rows, order and ``extra_info`` are preserved, so the output is a drop-in
replacement for the rebuild output consumed by ``mix_replay.py``.

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

    python3 examples/reasoning_rl/scripts/patch_code_parquet.py \
        --input  ~/data/reasoning_rl/code/train_code.parquet \
        --output ~/data/reasoning_rl/code_v2/train_code.parquet
"""

from __future__ import annotations

import argparse
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
    prompt = row["prompt"][0]["content"]
    if new_instruction not in prompt:
        if OLD_FUNCTION_INSTRUCTION in prompt:
            prompt = prompt.replace(OLD_FUNCTION_INSTRUCTION, new_instruction)
        else:
            prompt = prompt.rstrip() + "\n\n" + new_instruction
        row["prompt"][0]["content"] = prompt
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


def patch_dataset(input_path: str, output_path: str, dry_run: bool = False) -> dict:
    import datasets

    data = datasets.load_dataset("parquet", data_files=input_path, split="train").to_list()

    rows_with_fn = prompts_changed = outputs_unquoted = rows_touched = 0
    for row in data:
        changed, unquoted = patch_row(row)
        gt = row.get("reward_model", {}).get("ground_truth")
        try:
            has_fn = isinstance(gt, str) and "fn_name" in json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            has_fn = False
        rows_with_fn += int(has_fn)
        prompts_changed += int(changed)
        outputs_unquoted += unquoted
        rows_touched += int(changed or unquoted > 0)

    stats = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "total_rows": len(data),
        "call_based_rows": rows_with_fn,
        "prompts_updated": prompts_changed,
        "outputs_unquoted": outputs_unquoted,
        "rows_touched": rows_touched,
    }

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        datasets.Dataset.from_list(data).to_parquet(output_path)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Existing (pre-fix) train_code.parquet.")
    parser.add_argument("--output", required=True, help="Destination for the patched parquet.")
    parser.add_argument("--dry_run", action="store_true", help="Report counts without writing anything.")
    args = parser.parse_args(argv)

    if not os.path.isfile(os.path.expanduser(args.input)):
        raise SystemExit(f"[patch_code] input not found: {os.path.abspath(os.path.expanduser(args.input))}")

    stats = patch_dataset(os.path.expanduser(args.input), os.path.expanduser(args.output), dry_run=args.dry_run)
    print(f"[patch_code] {'DRY RUN' if args.dry_run else 'wrote ' + stats['output']}")
    for key in ("total_rows", "call_based_rows", "prompts_updated", "outputs_unquoted", "rows_touched"):
        print(f"[patch_code] {key}={stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
