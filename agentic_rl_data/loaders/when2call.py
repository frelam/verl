"""When2Call loader (abstain / ask-for-info decisions, plan §1.5 optional).

Verified raw path (GitHub README, 2026-11-05): ``NVIDIA/When2Call``
(Apache-2.0), HF id ``nvidia/When2Call`` named in the README.  Only the
**train** split is training-legal:

* ``data/train/when2call_train_sft.jsonl`` -- fields ``tools`` + ``messages``;
  assistant responses with tool calls are wrapped in ``<TOOLCALL> ... </TOOLCALL>``.
* the test split is derived from BFCL v2 Live -> must never be trained on;
* the train split is synthesized from ``Salesforce/xlam-function-calling-60k``,
  i.e. the same provenance as the existing ``apigen`` source, so exact /
  MinHash dedup against apigen is required.

Mapping: assistant response contains a tool call -> ``st_fc``; otherwise the
correct behaviour is abstain / ask for missing info -> ``abstain`` with
``gt_calls = []``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from agentic_rl_data.loaders.common import (
    NO_TOOL_SYSTEM,
    build_messages,
    build_reference,
    build_task,
    iter_source_records,
    normalize_tools,
    parse_tool_calls,
    record_skip,
    safe_task_id,
    unwrap_text,
)

logger = logging.getLogger(__name__)

SOURCE = "when2call"
LICENSE = "apache-2.0 (repo verified); HF card unreachable[UNVERIFIED]"
_DEFAULT_HF_DATASET = "nvidia/When2Call"


def _assistant_response(record: dict[str, Any]) -> str:
    """Last assistant response from the SFT record (or the pref chosen response)."""
    chosen = record.get("chosen_response")
    if isinstance(chosen, dict):
        return unwrap_text(chosen.get("content"))
    for key in ("messages", "conversations"):
        entries = record.get(key)
        if isinstance(entries, list):
            for entry in reversed(entries):
                if isinstance(entry, dict) and str(entry.get("role") or entry.get("from")) in (
                    "assistant",
                    "gpt",
                ):
                    return unwrap_text(entry.get("content") if "content" in entry else entry.get("value"))
    return ""


def _user_texts(record: dict[str, Any]) -> list[str]:
    chosen = record.get("chosen_response")
    texts: list[str] = []
    for key in ("messages", "conversations"):
        entries = record.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and str(entry.get("role") or entry.get("from")) in (
                    "user",
                    "human",
                ):
                    text = unwrap_text(entry.get("content") if "content" in entry else entry.get("value"))
                    if text.strip():
                        texts.append(text.strip())
    return texts


def load_tasks(cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``st_fc`` / ``abstain`` tasks from the When2Call train split."""
    cfg = dict(cfg or {})
    if not any(key in cfg for key in ("records", "path", "raw_path", "hf_dataset", "dataset")):
        cfg["hf_dataset"] = _DEFAULT_HF_DATASET
    max_samples = int(cfg.get("max_samples") or 0)
    license_name = cfg.get("license", LICENSE)
    emitted = 0
    for index, record in enumerate(iter_source_records(cfg, SOURCE)):
        if max_samples and emitted >= max_samples:
            return
        users = _user_texts(record)
        if not users:
            record_skip(cfg, "missing_user_messages")
            continue
        response = _assistant_response(record)
        if not response.strip():
            record_skip(cfg, "missing_assistant_response")
            continue
        calls = parse_tool_calls(response)
        task = build_task(
            source=SOURCE,
            task_id=safe_task_id(SOURCE, record.get("uuid") or record.get("id"), index),
            task_type="st_fc" if calls else "abstain",
            messages=build_messages(NO_TOOL_SYSTEM, users),
            tools=normalize_tools(record.get("tools")),
            env={"type": "none", "config": {"max_turns": 1}},
            reference=build_reference(gt_calls=[calls] if calls else []),
            max_turns=1,
            expected_turns=1,
            license=license_name,
            meta_extra={
                "raw_index": index,
                "split": cfg.get("split") or "train_sft",
                "group_id": str(record.get("uuid") or index),
            },
        )
        emitted += 1
        yield task


__all__ = ["SOURCE", "LICENSE", "load_tasks"]
