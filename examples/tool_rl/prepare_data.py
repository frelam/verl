#!/usr/bin/env python3
"""Download and prepare tool-use datasets for Qwen3-4B GRPO training in verl.

Ported from slime ``examples/tool_rl/data/download_data.py``. Downloads
APIGen, ToolACE, Hammer, BFCL and converts them to verl's parquet schema
used by ``verl.utils.dataset.RLHFDataset``.

Since the LLM-judge (RM) reward mode is not migrated, samples **without**
structured ground-truth tool calls are dropped by default
(``--keep-unlabeled`` disables the filter).

Output parquet columns
----------------------
- ``data_source``   : ``"tool_rl"``
- ``prompt``        : ``[{"role": ..., "content": ...}, ...]`` chat messages
                      (consumed via ``data.prompt_key=prompt``)
- ``tools``         : per-sample tool schemas
                      (consumed by the ``tool_rl_agent`` agent loop)
- ``reward_model``  : ``{"style": "rule", "ground_truth": <label str>}``
- ``extra_info``    : ``{"index", "task_id", "source", "tools",
                      "ground_truth_calls", ...}`` — ``tools`` and
                      ``ground_truth_calls`` are read by the reward function;
                      ``ground_truth_calls == []`` means "no tools needed".

Usage
-----
.. code-block:: bash

    python examples/tool_rl/prepare_data.py -o ./data/tool_rl
    python examples/tool_rl/prepare_data.py -o ./data/tool_rl --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Instruction — prepended to first user message
# ============================================================================

_INSTRUCTION = (
    "At no point should you assume any information about location, date, "
    "or any other details. Stay humble and honest. "
    "The entire task can be solved through multiple rounds of dialogue, "
    "gathering detailed information step by step — "
    "there is no need to solve everything in one go."
)

_DEFAULT_MAX = 5000
_SEED = 42
_DATA_SOURCE = "tool_rl"


# ============================================================================
# Helpers
# ============================================================================

def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool definitions to standard format."""
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        if not name:
            continue
        params = t.get("parameters", {})
        # Qwen chat template expects JSON Schema properties
        if isinstance(params, dict) and "properties" not in params:
            params = {"type": "object", "properties": params}
        result.append({
            "name": name,
            "description": t.get("description", ""),
            "parameters": params,
        })
    return result


def _format_gt(answers: list[dict[str, Any]]) -> str:
    """Format ground truth as readable string."""
    if not answers:
        return ""
    lines = []
    for a in answers:
        name = a.get("name", "")
        args = a.get("arguments", {}) or {}
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        else:
            args_str = str(args)
        lines.append(f"  {name}({args_str})")
    return "Ground truth:\n" + "\n".join(lines)


def _make_meta(source: str, task_id: str, tools: list, gt: Any, **extra) -> dict:
    return {
        "benchmark": "tool_rl",
        "source": source,
        "task_id": task_id,
        "ground_truth": gt,  # None=no label, []=no tools needed, [{...}]=tool calls
        "has_ground_truth": bool(gt),
        "tools": _normalize_tools(tools),
        "max_turns": 1,
        **extra,
    }


def _prepend_instruction(messages: list[dict]) -> list[dict]:
    """Prepend instruction to the first user message in the conversation."""
    for msg in messages:
        if msg.get("role") == "user":
            msg["content"] = (
                f"<instruction>\n{_INSTRUCTION}\n</instruction>\n\n"
                + msg["content"]
            )
            break
    return messages


# ============================================================================
# APIGen loader
# ============================================================================

def load_apigen(max_samples: int) -> list[dict[str, Any]]:
    """Load APIGen — single JSON file via hf_hub_download."""
    logger.info("Loading APIGen (Salesforce/xlam-function-calling-60k)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("pip install huggingface_hub")
        return []

    try:
        path = hf_hub_download(
            "Salesforce/xlam-function-calling-60k",
            "xlam_function_calling_60k.json",
            repo_type="dataset",
        )
    except Exception as e:
        logger.warning("APIGen download failed: %s", e)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for sample in data:
        if len(tasks) >= max_samples:
            break
        query = sample.get("query", "")
        if not query:
            continue

        tools_raw = sample.get("tools", "[]")
        answers_raw = sample.get("answers", "[]")
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except json.JSONDecodeError:
            tools = []
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except json.JSONDecodeError:
            answers = []

        messages = _prepend_instruction([
            {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed to answer user queries accurately."},
            {"role": "user", "content": query},
        ])

        tasks.append({
            "messages": messages,
            "tools": _normalize_tools(tools),
            "label": _format_gt(answers),
            "metadata": _make_meta("apigen", f"apigen-{sample.get('id', '?')}", tools, answers),
        })

    logger.info("APIGen: %d samples", len(tasks))
    return tasks


# ============================================================================
# ToolACE loader (multi-turn → split into single-turn)
# ============================================================================

def load_toolace(max_samples: int) -> list[dict[str, Any]]:
    """Load ToolACE — split multi-turn conversations into single-turn samples."""
    logger.info("Loading ToolACE (Team-ACE/ToolACE)...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        return []

    try:
        ds = load_dataset("Team-ACE/ToolACE", split="train")
    except Exception as e:
        logger.warning("ToolACE: %s", e)
        return []

    tasks = []
    for i, sample in enumerate(ds):
        if len(tasks) >= max_samples:
            break
        system = sample.get("system", "")
        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        tools = _extract_tools_from_text(system)
        history: list[dict] = []

        for ti, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = turn.get("from", "")
            value = str(turn.get("value", ""))

            if role == "user":
                messages = [{"role": "system", "content": system}]
                for h in history[-8:]:
                    messages.append(dict(h))
                messages.append({"role": "user", "content": value})
                messages = _prepend_instruction(messages)

                assistant_resp = _find_next_assistant(conversations, ti)
                gt_calls = _parse_qwen_tool_calls(assistant_resp)

                tasks.append({
                    "messages": messages,
                    "tools": _normalize_tools(tools),
                    "label": _format_gt(gt_calls) + (
                        f"\nReference:\n{assistant_resp[:1000]}"
                        if assistant_resp else ""
                    ),
                    "metadata": _make_meta(
                        "toolace", f"toolace-{i}-t{ti}", tools, gt_calls,
                        conversation_turn=ti,
                    ),
                })
                history.append({"role": "user", "content": value})

            elif role == "assistant":
                history.append({"role": "assistant", "content": value[:800]})
            elif role == "tool":
                history.append({"role": "tool", "content": value[:500]})

            if len(tasks) >= max_samples:
                break

    logger.info("ToolACE: %d single-turn samples", len(tasks))
    return tasks


def _find_next_assistant(conversations: list, idx: int) -> str:
    for j in range(idx + 1, len(conversations)):
        t = conversations[j]
        if isinstance(t, dict) and t.get("from") == "assistant":
            return str(t.get("value", ""))
    return ""


# ============================================================================
# Hammer loader
# ============================================================================

def load_hammer(max_samples: int) -> list[dict[str, Any]]:
    """Load Hammer irrelevance data."""
    logger.info("Loading Hammer (MadeAgents/xlam-irrelevance-7.5k)...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        return []

    try:
        ds = load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")
    except Exception as e:
        logger.warning("Hammer: %s", e)
        return []

    tasks = []
    for i, sample in enumerate(ds):
        if len(tasks) >= max_samples:
            break
        query = sample.get("query", "")
        if not query:
            continue
        tools_raw = sample.get("tools", "[]")
        answers_raw = sample.get("answers", "[]")
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except json.JSONDecodeError:
            tools = []
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except json.JSONDecodeError:
            answers = []

        messages = _prepend_instruction([
            {"role": "system", "content": "You are a helpful assistant. Determine if tools are needed for the user's request."},
            {"role": "user", "content": query},
        ])

        tasks.append({
            "messages": messages,
            "tools": _normalize_tools(tools),
            "label": _format_gt(answers),
            "metadata": _make_meta("hammer", f"hammer-{i}", tools, answers,
                                   is_irrelevant=not bool(answers)),
        })

    logger.info("Hammer: %d samples", len(tasks))
    return tasks


# ============================================================================
# BFCL loader
# ============================================================================

def load_bfcl(max_samples: int) -> list[dict[str, Any]]:
    """Load BFCL — split multi-turn, keep single-turn."""
    logger.info("Loading BFCL (gorilla-llm/Berkeley-Function-Calling-Leaderboard)...")
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        logger.error("pip install huggingface_hub")
        return []

    try:
        files = list_repo_files(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard", repo_type="dataset",
        )
    except Exception as e:
        logger.warning("BFCL: %s", e)
        return []

    json_files = [f for f in files if f.endswith(".json")]
    priority = ["simple", "multiple", "parallel", "multi_turn"]
    json_files.sort(key=lambda f: (not any(p in f.lower() for p in priority), f))

    tasks = []
    for jf in json_files:
        if len(tasks) >= max_samples:
            break
        try:
            path = hf_hub_download(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                jf, repo_type="dataset",
            )
        except Exception:
            continue
        category = jf.replace(".json", "").replace("BFCL_v3_", "")
        with open(path) as f:
            for line in f:
                if len(tasks) >= max_samples:
                    break
                try:
                    raw = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                tasks.extend(_parse_bfcl(raw, category))

    logger.info("BFCL: %d samples", len(tasks))
    return tasks


def _extract_bfcl_turns(question: Any) -> list[list[dict]] | None:
    """Extract turns from BFCL v3 question field.

    BFCL v3 format: ``[[{"role": "user", "content": "..."}], ...]``
    Each outer element is a turn with one or more messages.
    Returns list of messages per turn (list of lists of dicts).
    """
    if isinstance(question, list) and len(question) > 0:
        # BFCL v3: list of turns, each turn is a list of messages
        if isinstance(question[0], list):
            return question
        # Some files have list of dicts (single turn)
        if isinstance(question[0], dict) and "role" in question[0]:
            return [question]
    return None


def _extract_text_from_bfcl_messages(messages: list) -> str:
    """Extract user text from BFCL message list."""
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts) if parts else ""


def _parse_bfcl(raw: dict, category: str) -> list[dict]:
    """Parse BFCL sample → single-turn samples."""
    funcs = raw.get("function") or raw.get("functions") or []
    if isinstance(funcs, dict):
        funcs = [funcs]
    tools = _normalize_tools(funcs)
    tid = raw.get("id", f"bfcl-{category}")

    # Skip auxiliary files
    if category.startswith("possible_answer/") or category.startswith("multi_turn_func_doc/"):
        return []

    turns_data = _extract_bfcl_turns(raw.get("question"))
    if turns_data is None:
        return []

    results = []
    history: list[dict] = []

    for ti, turn_msgs in enumerate(turns_data):
        if not isinstance(turn_msgs, list):
            continue
        query_text = _extract_text_from_bfcl_messages(turn_msgs)
        if not query_text:
            continue

        msgs = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
        ]
        for h in history[-8:]:
            msgs.append(dict(h))
        msgs.append({"role": "user", "content": query_text})
        msgs = _prepend_instruction(msgs)

        # Parse ground truth for this turn
        gt_raw = raw.get("ground_truth") or raw.get("answers") or raw.get("answer") or ""
        if isinstance(gt_raw, list):
            gt_str = "\n".join(
                json.dumps(g, ensure_ascii=False) for g in gt_raw if g
            ) if gt_raw else ""
        elif isinstance(gt_raw, str):
            gt_str = gt_raw
        else:
            gt_str = str(gt_raw) if gt_raw else ""
        if gt_str.strip():
            gt = _parse_qwen_tool_calls(gt_str)
        else:
            gt = None  # BFCL has no ground truth — needs RM mode (not migrated)

        results.append({
            "messages": msgs,
            "tools": tools,
            "label": _format_gt(gt) + (f"\nReference:\n{gt_str[:800]}" if gt_str.strip() else ""),
            "metadata": _make_meta(
                f"bfcl/{category}", f"{tid}-t{ti}", tools, gt,
                bfcl_category=category,
            ),
        })

        # Add to history for multi-turn context
        history.append({"role": "user", "content": query_text})
        # Look for assistant response in the same turn
        for msg in turn_msgs:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    history.append({"role": "assistant", "content": content[:500]})

    return results


# ============================================================================
# Tool extraction
# ============================================================================

def _extract_tools_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool definitions from system prompt text."""
    tools = []
    # Try to find the outermost JSON array of tools in the text
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        try:
            candidates = json.loads(array_match.group(0))
            if isinstance(candidates, list):
                for c in candidates:
                    if isinstance(c, dict) and "name" in c:
                        tools.append(c)
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: try extracting individual objects with nested braces
    if not tools:
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    block = text[start:i+1]
                    try:
                        obj = json.loads(block)
                        if isinstance(obj, dict) and "name" in obj:
                            tools.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return tools


def _parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen XML tool calls from assistant response.

    Format::

        <tool_call>
        <function=name>
        <parameter=param>
        value
        </parameter>
        </function>
        </tool_call>
    """
    calls = []
    for tc_match in re.finditer(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE,
    ):
        block = tc_match.group(1)
        # Parse function name
        func_match = re.search(r"<function=(\w[\w.]*)>", block)
        if not func_match:
            continue
        func_name = func_match.group(1)

        # Parse parameters
        args = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL,
        ):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            # Try JSON parse for structured values
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pname] = pval

        calls.append({"name": func_name, "arguments": args})

    # Fallback: JSON tool calls — bracket-matching handles nested args
    if not calls:
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and "name" in obj:
                            calls.append(obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    start = -1

    return calls


# ============================================================================
# Data augmentation — negative samples & robustness perturbations
# ============================================================================
#
# Motivation: the raw datasets are (almost) all positive samples — the label
# always contains the "correct" tool calls, which encourages shortcut learning
# (memorising parameter names, copying schema defaults, blindly calling a
# tool whose *name* matches the query).  To counter this, a fraction of the
# samples with ground-truth tool calls is perturbed in one of four ways:
#
#   1a. tool_rename     — rename a label tool in the prompt (description
#                         unchanged); the label follows the new name.
#                         Teaches: bind to the description, not the name.
#   1b. desc_replace    — replace a label tool's description with an
#                         unrelated one; the label becomes empty (no tool
#                         should be called).  This is the negative sample:
#                         match_tool_calls_against_label() already penalises
#                         spurious calls when the label is empty.
#   2.  param_rename    — rename parameter names in the tool schema; the
#                         label's argument keys follow (values unchanged).
#                         Teaches: read parameter descriptions, don't
#                         memorise canonical parameter names.
#   3.  default_shuffle — randomise ``default`` values in the tool schema.
#                         Teaches: don't copy schema defaults into calls.
#
# Each augmented sample gets ``metadata["augmented"] = <strategy>``.

# Unrelated tool names used for renames — clearly off-topic for typical
# function-calling queries, but with the original description kept they
# remain the "correct" tool under a new name.
_IRRELEVANT_TOOL_NAMES = [
    "blend_smoothie_recipe",
    "translate_morse_code",
    "calculate_mortgage_rate",
    "compose_haiku_poem",
    "render_star_chart",
    "tune_guitar_strings",
    "estimate_paint_coverage",
    "decode_vin_number",
]

# Unrelated descriptions used for desc_replace — after the swap the tool is
# no longer suitable for the query, so the correct behaviour is *no* call.
_IRRELEVANT_DESCRIPTIONS = [
    "Render a 3D animation of a rotating geometric shape.",
    "Compose a short poem about the changing seasons.",
    "Estimate the calorie count of a dish from its photo.",
    "Generate chord progressions for a given musical key.",
    "Simulate the orbit of a satellite around a planet.",
    "Design a knitting pattern for a winter scarf.",
]

# Generic parameter names used for param_rename.
_GENERIC_PARAM_NAMES = [
    "input_value",
    "query_text",
    "target_item",
    "config_option",
    "content_body",
    "request_field",
    "user_option",
    "item_reference",
]


def _augment_tool_copies(task: dict, name: str):
    """Yield every schema copy of tool ``name`` (top-level + metadata).

    Tasks carry two independent normalised copies of the tool list —
    ``task["tools"]`` and ``task["metadata"]["tools"]``.  Both must stay in
    sync under any mutation.
    """
    seen = set()
    for tool_list in (task.get("tools"), task.get("metadata", {}).get("tools")):
        if not isinstance(tool_list, list):
            continue
        for t in tool_list:
            if isinstance(t, dict) and t.get("name") == name and id(t) not in seen:
                seen.add(id(t))
                yield t


def _label_tool_candidates(task: dict) -> list[str]:
    """Names of tools that appear both in the label and in the prompt."""
    gt = task.get("metadata", {}).get("ground_truth")
    if not isinstance(gt, list) or not gt:
        return []
    label_names = {c.get("name") for c in gt if isinstance(c, dict) and c.get("name")}
    prompt_names = {
        t.get("name")
        for t in (task.get("tools") or [])
        if isinstance(t, dict) and t.get("name")
    }
    return sorted(label_names & prompt_names)


def _rename_in_text(text: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", new, text)


def _rename_in_messages(messages: list[dict], old: str, new: str) -> None:
    """Keep embedded tool references (system prompt / history) consistent."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and old in content:
            m["content"] = _rename_in_text(content, old, new)


def _augment_tool_rename(task: dict, rng: random.Random) -> str | None:
    """Strategy 1a: rename a label tool in the prompt; label follows."""
    candidates = _label_tool_candidates(task)
    if not candidates:
        return None
    old = rng.choice(candidates)
    existing = {
        t.get("name") for t in (task.get("tools") or []) if isinstance(t, dict)
    }
    pool = [n for n in _IRRELEVANT_TOOL_NAMES if n not in existing]
    if not pool:
        return None
    new = rng.choice(pool)

    for t in _augment_tool_copies(task, old):
        t["name"] = new
    gt = task["metadata"]["ground_truth"]
    for call in gt:
        if isinstance(call, dict) and call.get("name") == old:
            call["name"] = new
    _rename_in_messages(task.get("messages") or [], old, new)
    if isinstance(task.get("label"), str) and old in task["label"]:
        task["label"] = _rename_in_text(task["label"], old, new)

    task["metadata"]["augmented"] = "tool_rename"
    task["metadata"]["augment_detail"] = {"old": old, "new": new}
    return "tool_rename"


def _augment_desc_replace(task: dict, rng: random.Random) -> str | None:
    """Strategy 1b: swap a label tool's description for an unrelated one.

    After the swap no declared tool fits the query, so this becomes a
    **negative sample**: ground_truth is emptied (``[]``, not ``None`` —
    the reward distinguishes "label says no tools" from "no label").
    """
    candidates = _label_tool_candidates(task)
    if not candidates:
        return None
    name = rng.choice(candidates)
    new_desc = rng.choice(_IRRELEVANT_DESCRIPTIONS)
    for t in _augment_tool_copies(task, name):
        t["description"] = new_desc

    task["label"] = ""
    task["metadata"]["ground_truth"] = []
    task["metadata"]["has_ground_truth"] = False
    task["metadata"]["augmented"] = "desc_replace"
    task["metadata"]["augment_detail"] = {"tool": name}
    return "desc_replace"


def _augment_param_rename(task: dict, rng: random.Random) -> str | None:
    """Strategy 2: rename schema parameter names; label keys follow."""
    candidates = _label_tool_candidates(task)
    if not candidates:
        return None
    rng.shuffle(candidates)

    for name in candidates:
        # Collect the union of schema properties across both copies.
        props: dict[str, Any] = {}
        for t in _augment_tool_copies(task, name):
            params = t.get("parameters")
            if isinstance(params, dict) and isinstance(params.get("properties"), dict):
                props.update(params["properties"])
        if not props:
            continue

        gt = task["metadata"]["ground_truth"]
        gt_arg_keys = {
            k
            for c in gt
            if isinstance(c, dict) and c.get("name") == name
            for k in (c.get("arguments") or {})
        }
        # Prefer renaming a parameter that the label actually uses.
        choices = sorted(gt_arg_keys & set(props)) or sorted(props)
        old = rng.choice(choices)
        pool = [p for p in _GENERIC_PARAM_NAMES if p not in props]
        if not pool:
            continue
        new = rng.choice(pool)

        for t in _augment_tool_copies(task, name):
            params = t.get("parameters")
            if not isinstance(params, dict):
                continue
            p = params.get("properties")
            if isinstance(p, dict) and old in p:
                p[new] = p.pop(old)
            req = params.get("required")
            if isinstance(req, list):
                params["required"] = [new if r == old else r for r in req]

        for call in gt:
            if isinstance(call, dict) and call.get("name") == name:
                args = call.get("arguments")
                if isinstance(args, dict) and old in args:
                    args[new] = args.pop(old)

        if isinstance(task.get("label"), str) and old in task["label"]:
            task["label"] = _rename_in_text(task["label"], old, new)

        task["metadata"]["augmented"] = "param_rename"
        task["metadata"]["augment_detail"] = {"tool": name, "old": old, "new": new}
        return "param_rename"
    return None


def _random_default_value(old: Any, ptype: str, rng: random.Random) -> Any:
    """Random replacement for a schema ``default``, kept type-compatible."""
    for _ in range(8):
        if ptype == "integer":
            v: Any = rng.randint(-1000, 1000)
        elif ptype == "number":
            v = round(rng.uniform(-1000, 1000), 2)
        elif ptype == "boolean":
            v = not old if isinstance(old, bool) else rng.random() < 0.5
        elif ptype == "array":
            v = []
        elif ptype == "object":
            v = {}
        else:  # string / unknown
            v = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
        if v != old:
            return v
    return None


def _augment_default_shuffle(task: dict, rng: random.Random) -> str | None:
    """Strategy 3: randomise ``default`` values in the tool schema."""
    changed: list[str] = []
    for tool_list in (task.get("tools"), task.get("metadata", {}).get("tools")):
        if not isinstance(tool_list, list):
            continue
        for t in tool_list:
            if not isinstance(t, dict):
                continue
            params = t.get("parameters")
            if not isinstance(params, dict) or not isinstance(params.get("properties"), dict):
                continue
            for pinfo in params["properties"].values():
                if not isinstance(pinfo, dict) or "default" not in pinfo:
                    continue
                v = _random_default_value(
                    pinfo["default"], str(pinfo.get("type", "string")), rng,
                )
                if v is not None:
                    pinfo["default"] = v
                    if t.get("name") not in changed:
                        changed.append(t.get("name"))

    if not changed:
        return None
    task["metadata"]["augmented"] = "default_shuffle"
    task["metadata"]["augment_detail"] = {"tools": changed}
    return "default_shuffle"


_AUGMENT_STRATEGIES = (
    _augment_tool_rename,
    _augment_desc_replace,
    _augment_param_rename,
    _augment_default_shuffle,
)


def augment_tasks(
    tasks: list[dict[str, Any]],
    ratio: float,
    rng: random.Random,
) -> int:
    """Perturb a ``ratio`` fraction of positive samples (label has tool calls).

    Each selected sample is mutated in place by the first applicable
    strategy from a shuffled order.  Returns the number of augmented tasks.
    """
    if ratio <= 0 or not tasks:
        return 0

    eligible = [
        i
        for i, t in enumerate(tasks)
        if isinstance(t.get("metadata", {}).get("ground_truth"), list)
        and t["metadata"]["ground_truth"]
    ]
    rng.shuffle(eligible)
    target = max(1, round(len(tasks) * ratio))

    augmented = 0
    for i in eligible[:target]:
        strategies = list(_AUGMENT_STRATEGIES)
        rng.shuffle(strategies)
        for strategy in strategies:
            if strategy(tasks[i], rng) is not None:
                augmented += 1
                break
    return augmented


# ============================================================================
# Validation
# ============================================================================

def validate_tasks(tasks: list) -> list:
    valid = []
    for t in tasks:
        msgs = t.get("messages", [])
        if not msgs:
            continue
        tools = t.get("tools", t.get("metadata", {}).get("tools", []))
        if not tools:
            continue
        user_content = next(
            (m["content"] for m in msgs if m.get("role") == "user"), "",
        )
        if len(user_content) > 65536:
            continue
        valid.append(t)
    removed = len(tasks) - len(valid)
    if removed:
        logger.info("Filtered %d invalid (%d remaining)", removed, len(valid))
    return valid


# ============================================================================
# verl schema conversion
# ============================================================================

def to_verl_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert slime task dicts to verl RLHFDataset rows."""
    rows = []
    for i, t in enumerate(tasks):
        meta = t["metadata"]
        tools = t.get("tools") or meta.get("tools") or []
        rows.append({
            "data_source": _DATA_SOURCE,
            "prompt": t["messages"],
            "tools": tools,
            "reward_model": {"style": "rule", "ground_truth": t.get("label", "")},
            "extra_info": {
                "index": i,
                "task_id": meta.get("task_id", f"task-{i}"),
                "source": meta.get("source", "unknown"),
                "tools": tools,
                "ground_truth_calls": meta.get("ground_truth"),
                "augmented": meta.get("augmented", ""),
            },
        })
    return rows


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows → %s", len(df), path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download tool-use datasets for Qwen3-4B RL (verl)")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-samples", type=int, default=_DEFAULT_MAX)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--val-samples", type=int, default=256,
                        help="Number of samples held out for validation parquet.")
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep samples without structured ground truth (they would need "
        "the RM reward mode, which is not migrated — by default they are dropped).",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.15,
        help="Fraction of positive samples to perturb for robustness "
        "(tool/param renames, negative samples, default shuffle). 0 disables.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = (["apigen", "toolace", "hammer", "bfcl"] if args.datasets == "all"
             else [n.strip() for n in args.datasets.split(",")])

    loaders = {"apigen": load_apigen, "toolace": load_toolace,
               "hammer": load_hammer, "bfcl": load_bfcl}

    all_tasks = []
    for name in names:
        if name not in loaders:
            logger.warning("Unknown %r, available: %s", name, sorted(loaders))
            continue
        tasks = loaders[name](args.max_samples)
        tasks = validate_tasks(tasks)
        if tasks and args.augment_ratio > 0:
            n_aug = augment_tasks(
                tasks, args.augment_ratio, random.Random(args.seed),
            )
            logger.info("Augmented %d/%d %s samples", n_aug, len(tasks), name)
        if tasks:
            all_tasks.extend(tasks)
        else:
            logger.warning("No samples for %s", name)

    if not all_tasks:
        logger.error("No datasets loaded.")
        sys.exit(1)

    # RM mode is not migrated: drop samples without structured ground truth.
    if not args.keep_unlabeled:
        before = len(all_tasks)
        all_tasks = [
            t for t in all_tasks
            if t.get("metadata", {}).get("ground_truth") is not None
        ]
        dropped = before - len(all_tasks)
        if dropped:
            logger.info(
                "Dropped %d unlabeled samples (RM mode not migrated; "
                "--keep-unlabeled to disable)", dropped,
            )

    if not all_tasks:
        logger.error("No labeled samples left after filtering.")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(all_tasks)

    n_val = min(args.val_samples, max(0, len(all_tasks) // 10))
    val_tasks = all_tasks[:n_val]
    train_tasks = all_tasks[n_val:]

    write_parquet(to_verl_rows(train_tasks), output_dir / "train.parquet")
    if val_tasks:
        write_parquet(to_verl_rows(val_tasks), output_dir / "val.parquet")

    logger.info("Done! train=%d val=%d → %s", len(train_tasks), len(val_tasks), output_dir)
    for src in sorted(set(t["metadata"]["source"] for t in all_tasks)):
        logger.info(
            "  %s: %d", src,
            sum(1 for t in all_tasks if t["metadata"]["source"] == src),
        )


if __name__ == "__main__":
    main()
