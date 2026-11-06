"""Code-execution loader (write -> run -> read tests -> fix, plan §1.4).

Covers the three viable dataset families behind one ``code_exec`` loader; the
task type is frozen to ``code_exec`` and the env is ``code_tests``:

* ``kind="apps"`` -- APPS fields ``question`` / ``input_output`` (JSON string
  with ``{"inputs": [...], "outputs": [...]}``) / ``solutions``;
  repo verified MIT (``hendrycks/apps``), HF card ``codeparrot/apps`` unverified.
* ``kind="taco"`` -- TACO fields ``question`` / ``input_output`` / ``solutions``;
  repo verified Apache-2.0 (``FlagOpen/TACO``), HF card ``BAAI/TACO`` unverified.
* ``kind="code_contests"`` -- CodeContests fields ``description`` /
  ``public_tests`` / ``private_tests`` (``{"input": [...], "output": [...]}``);
  repo verified Apache-2.0 and archived (``google-deepmind/code_contests``).

Reward source is the test cases in ``reference["unit_tests"]`` (JSON string:
``{"kind", "inputs", "outputs"}``); the standard solution is deliberately not
stored in the task.  Repository-level (SWE-style) tasks are out of scope
(plan §1.4: they exceed the turn budget).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    build_messages,
    build_reference,
    build_task,
    env_code_tests,
    first_non_empty,
    iter_source_records,
    parse_json_maybe,
    record_skip,
    safe_task_id,
)

logger = logging.getLogger(__name__)

SOURCE = "code_exec"
LICENSE = "mixed: apps=MIT, taco=Apache-2.0, code_contests=Apache-2.0 (repo-level)"
_HF_DATASETS = {
    "apps": "codeparrot/apps",
    "taco": "BAAI/TACO",
    "code_contests": "deepmind/code_contests",
}


def _tests_payload(kind: str, record: dict[str, Any]) -> dict[str, Any] | None:
    if kind in ("apps", "taco"):
        input_output = parse_json_maybe(record.get("input_output"))
        if isinstance(input_output, dict):
            inputs = input_output.get("inputs")
            outputs = input_output.get("outputs")
            if isinstance(inputs, list) and isinstance(outputs, list) and inputs:
                return {"kind": kind, "inputs": inputs, "outputs": outputs}
        return None
    # code_contests
    public = parse_json_maybe(record.get("public_tests")) or {}
    private = parse_json_maybe(record.get("private_tests")) or {}
    inputs: list[Any] = []
    outputs: list[Any] = []
    for source in (public, private):
        if isinstance(source, dict):
            inputs.extend(source.get("input") or [])
            outputs.extend(source.get("output") or [])
    if inputs and outputs:
        return {"kind": kind, "inputs": inputs, "outputs": outputs}
    return None


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``code_exec`` tasks whose unit tests live in ``unit_tests``."""
    cfg = dict(cfg or {})
    kind = str(cfg.get("kind") or "apps").strip().lower()
    if kind not in _HF_DATASETS:
        raise ValueError(f"{SOURCE}: cfg['kind'] must be one of {sorted(_HF_DATASETS)}")
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _HF_DATASETS[kind]
    max_samples = int(cfg.get("max_samples") or 0)
    max_turns = int(cfg.get("max_turns") or 15)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        question = first_non_empty(
            record.get("question"), record.get("description"), record.get("problem")
        )
        if not isinstance(question, str) or not question.strip():
            record_skip(cfg, "missing_question")
            continue
        tests = _tests_payload(kind, record)
        if tests is None:
            logger.debug("%s: record %d has no usable tests", SOURCE, index)
            record_skip(cfg, "no_usable_tests")
            continue
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("id") or record.get("name"), index),
            task_type="code_exec",
            messages=build_messages(
                "You are a competitive programmer. Write a Python solution. Tests will be executed; "
                "read the failure output and fix the code. Final code goes in a single ```python block.",
                [question],
            ),
            tools=[],
            env=env_code_tests(max_turns=max_turns, redact=list(cfg.get("redact") or [])),
            reference=build_reference(
                unit_tests=json.dumps(tests, ensure_ascii=False, sort_keys=True),
                gt_calls=[],
            ),
            max_turns=max_turns,
            expected_turns=cfg.get("expected_turns") or None,
            difficulty=record.get("difficulty") if isinstance(record.get("difficulty"), (int, float)) else None,
            license=license_name,
            meta_extra={"raw_index": index, "kind": kind, "tests": len(tests["inputs"])},
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
