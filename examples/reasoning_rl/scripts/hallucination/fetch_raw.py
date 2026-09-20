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
"""Download every raw file the hallucination build needs (design doc section 4).

One-shot and idempotent: a file that already exists with a non-zero size is left
alone unless ``--force`` is passed, so re-running after a partial failure only
fetches what is missing.  Every completed download is recorded in
``raw_manifest.json`` next to the data with its URL, byte count and sha256, which
is what makes a later build reproducible and lets :mod:`verify_*` scripts state
which bytes they audited.

The pool is the sources of design doc section 4.8 table B; D25 removed the one
source whose judgement signal carried no verifiable answer layer, so nothing is
fetched for it here.

Usage::

    /lhy/miniconda3/envs/lhy/bin/python fetch_raw.py                       # everything
    /lhy/miniconda3/envs/lhy/bin/python fetch_raw.py --only kk,gsmic
    /lhy/miniconda3/envs/lhy/bin/python fetch_raw.py --dest /data/halluc/raw

Licence note: each source carries its upstream licence in ``FileSpec.license``.
They are *not* uniform -- UMWP is CC-BY-SA-4.0 (share-alike, and the repo ships
no LICENSE file at all) while CREPE is BSD.  Anything redistributed downstream
has to reconcile those, so the field is recorded here rather than left in a
README somewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DEST = os.path.expanduser("~/data/reasoning_rl/halluc/raw")

# Hidden Service-style config: the datasets-server parquet refs are the only
# stable way to fetch KK/CREPE/SUM, because `resolve/main` serves a different
# (losing) layout for those repos.
_HF = "https://huggingface.co/datasets"
_KK_CLEAN = f"{_HF}/K-and-K/knights-and-knaves/resolve/refs%2Fconvert%2Fparquet"
_KK_PERT = f"{_HF}/K-and-K/perturbed-knights-and-knaves/resolve/refs%2Fconvert%2Fparquet"
_KK_PERTURBATIONS = (
    "flip_role",
    "perturbed_leaf",
    "perturbed_statement",
    "random_pair",
    "reorder_statement",
    "uncommon_name",
)
_KK_SIZES = (2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class FileSpec:
    """One downloadable artifact.

    ``name`` is the local filename; ``subdir`` is the directory under the
    destination root.  One ``subdir`` per source mirrors the recon layout the
    reports were written against.
    """

    subdir: str
    name: str
    url: str
    license: str = ""
    note: str = ""
    optional: bool = False


def _kk_specs() -> list[FileSpec]:
    specs = []
    for split in ("train", "test"):
        for n in _KK_SIZES:
            specs.append(
                FileSpec(
                    subdir="kk",
                    name=f"clean__{split}__{n}ppl.parquet",
                    url=f"{_KK_CLEAN}/{split}/{n}ppl/0000.parquet",
                    license="CC-BY-4.0",
                    note=f"K&K clean puzzle, {n} inhabitants",
                )
            )
    for split in ("train", "test"):
        for perturbation in _KK_PERTURBATIONS:
            specs.append(
                FileSpec(
                    subdir="kk",
                    name=f"perturbed__{split}__{perturbation}.parquet",
                    url=f"{_KK_PERT}/{split}/{perturbation}/0000.parquet",
                    license="CC-BY-4.0",
                    note=f"K&K perturbation family {perturbation}",
                )
            )
    return specs


FILES: list[FileSpec] = [
    # --- GSM-IC: the solvable-only anchor (D11) and the D17 template engine ---
    FileSpec("gsmic", "GSM-IC_2step.json", "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_2step.json", "MIT", "34,220 rows"),
    FileSpec("gsmic", "GSM-IC_mstep.json", "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_mstep.json", "MIT", "23,832 rows"),
    # gsm8k is fetched only so the adapter can re-certify GSM-IC's `answer`
    # against the original ground truth (the recon measured 58,052/58,052 agree).
    FileSpec("gsmic", "gsm8k_train.jsonl", "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl", "MIT", "GSM8K reference"),
    FileSpec("gsmic", "gsm8k_test.jsonl", "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl", "MIT", "GSM8K reference"),
    FileSpec("gsmic", "gsm8k_validation.jsonl", "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/gsm8k_validation.jsonl", "MIT", "GSM8K validation subset of GSM-IC"),
    # --- MiP ---
    FileSpec("mip", "gsm8k.json", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/main/data/gsm8k.json", "MIT", "582 rows, has in-file original"),
    FileSpec("mip", "math.json", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/main/data/math.json", "MIT", "52 rows, LaTeX answers"),
    FileSpec("mip", "svamp.json", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/main/data/svamp.json", "MIT", "300 rows, cross-problem splice, no original"),
    FileSpec("mip", "formula.json", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/main/data/formula.json", "MIT", "50 rows, DISCARDED (no answer, no question)", optional=True),
    FileSpec("mip", "SVAMP.json", "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json", "MIT", "public SVAMP, used only to test the splice claim"),
    # --- UMWP ---
    FileSpec("umwp", "StandardDataset.jsonl", "https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/main/data/StandardDataset.jsonl", "CC-BY-SA-4.0", "5,200 rows = 2,600 pairs; repo ships NO LICENSE file"),
    FileSpec("umwp", "README.md", "https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/main/README.md", "CC-BY-SA-4.0", "licence is asserted here only", optional=True),
    FileSpec("umwp", "StandardDataset.py", "https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/main/StandardDataset.py", "Apache-2.0", "upstream loader", optional=True),
    # --- FalseQA ---
    FileSpec("falseqa", "train.csv", "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/train.csv", "Apache-2.0", "strict 50:50 per split"),
    FileSpec("falseqa", "valid.csv", "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/valid.csv", "Apache-2.0", "strict 50:50 per split"),
    FileSpec("falseqa", "test.csv", "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/test.csv", "Apache-2.0", "answers are Python list strings"),
    # --- SUM ---
    FileSpec("sum", "train.parquet", f"{_HF}/lime-nlp/Synthetic_Unanswerable_Math/resolve/refs%2Fconvert%2Fparquet/synthetic_unanswerable_math/train/0000.parquet", "Apache-2.0", "synthetic unanswerable math"),
    FileSpec("sum", "test.parquet", f"{_HF}/lime-nlp/Synthetic_Unanswerable_Math/resolve/refs%2Fconvert%2Fparquet/synthetic_unanswerable_math/test/0000.parquet", "Apache-2.0", "synthetic unanswerable math"),
    # --- CREPE ---
    FileSpec("crepe", "crepe_train.parquet", f"{_HF}/tasksource/CREPE/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", "BSD", "3,462 rows"),
    FileSpec("crepe", "crepe_validation.parquet", f"{_HF}/tasksource/CREPE/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet", "BSD", "2,000 rows (upstream calls it dev)"),
    FileSpec("crepe", "crepe_test.parquet", f"{_HF}/tasksource/CREPE/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet", "BSD", "3,004 rows"),
] + _kk_specs()

# TreeCut is not listed above: its HF mirror (`jouyang/treecut-math`) has no
# datasets-server parquet refs, so its file paths have to come from an
# enumeration of `/tree/main` rather than a fixed layout.  Its adapter agent adds
# the concrete specs here; until then `--only treecut` reports the known subdirs
# and exits 2 rather than silently fetching nothing.
PENDING_SUBDIRS = ("treecut",)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: FileSpec, dest_root: Path, force: bool = False, retries: int = 3) -> dict:
    """Fetch one spec, returning its manifest entry.

    A zero-byte file counts as missing: some of these endpoints answer 200 with an
    empty body for a path that does not exist, and treating that as success would
    silently poison every downstream adapter.
    """
    target = dest_root / spec.subdir / spec.name
    entry = {
        "subdir": spec.subdir,
        "name": spec.name,
        "url": spec.url,
        "license": spec.license,
        "note": spec.note,
        "path": str(target),
    }
    if target.exists() and target.stat().st_size > 0 and not force:
        entry.update(status="cached", bytes=target.stat().st_size, sha256=sha256_of(target))
        return entry

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(spec.url, headers={"User-Agent": "verl-halluc-fetch/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries - 1:
                entry.update(status="failed", error=str(exc))
                return entry
            time.sleep(2**attempt)

    size = target.stat().st_size
    if size == 0:
        entry.update(status="failed", error="empty body", cached_from=last_error)
        return entry
    entry.update(status="ok", bytes=size, sha256=sha256_of(target))
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=DEFAULT_DEST, help="raw data root")
    parser.add_argument("--only", default=None, help="comma-separated subdirs, e.g. kk,gsmic")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    dest_root = Path(args.dest)
    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    specs = [s for s in FILES if wanted is None or s.subdir in wanted]
    if not specs:
        known = sorted({s.subdir for s in FILES} | set(PENDING_SUBDIRS))
        print(f"no files match --only {args.only!r}; known subdirs: {known}")
        return 2

    entries, failures = [], []
    for spec in specs:
        entry = download(spec, dest_root, force=args.force)
        entries.append(entry)
        marker = {"ok": "OK  ", "cached": "have", "failed": "FAIL"}[entry["status"]]
        size = entry.get("bytes", 0)
        print(f"{marker} {spec.subdir}/{spec.name:<40} {size/1e6:8.2f} MB  {spec.note}")
        if entry["status"] == "failed" and not spec.optional:
            failures.append(entry)

    manifest_path = dest_root / "raw_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"files": entries}, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nmanifest -> {manifest_path}  ({len(entries)} files, {len(failures)} hard failures)")

    if failures:
        print("\nREQUIRED FILES FAILED TO DOWNLOAD -- the build cannot proceed:", file=sys.stderr)
        for entry in failures:
            print(f"  {entry['url']}: {entry['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
