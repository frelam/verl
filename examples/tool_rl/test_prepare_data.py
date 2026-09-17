"""Tests for tool_rl data preparation — schema isolation.

``_normalize_tools`` used to hand the *same* ``parameters`` object to every
sample that declared a tool (the Seal-Tools / API-Bank loaders share one
``schemas`` pool).  ``_augment_param_rename`` mutates that object in place, so
a rename performed for one sample silently rewrote every other sample's
declared schema while leaving *their* labels untouched: 331/5000 Seal-Tools
rows ended up with a label parameter key that no longer existed in the prompt,
i.e. unwinnable (a schema-conformant call caps at Dim1 = 0.5).

Sibling tools that share a description stay in the menu on purpose — the
reward accepts either (see ``test_tool_rl_reward.py``).

Run from the repo root:

.. code-block:: bash

    python -m pytest examples/tool_rl/test_prepare_data.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl import prepare_data  # noqa: E402
from examples.tool_rl.prepare_data import (  # noqa: E402
    _format_gt,
    _make_meta,
    _normalize_tools,
)

_MATCH_TOOL = {
    "name": "getMatchInfo",
    "description": "Retrieve information about a football match",
    "parameters": {
        "type": "object",
        "properties": {"match_id": {"type": "string", "description": "The match id"}},
        "required": ["match_id"],
    },
}


def _task_for(shared_schema: dict, task_id: str, name: str, arguments: dict) -> dict:
    """Build a task the way ``load_sealtools`` does, from a shared schema.

    The ground truth is rebuilt per task on purpose: production loaders build
    one ``gt`` list per sample, and ``_augment_param_rename`` mutates the
    argument dicts in place.
    """
    gt = [{"name": name, "arguments": dict(arguments)}]
    tools = _normalize_tools([shared_schema])
    return {
        "messages": [{"role": "user", "content": "q"}],
        "tools": tools,
        "label": _format_gt(gt),
        "metadata": _make_meta("test", task_id, tools, gt),
    }


def test_normalize_tools_returns_independent_copies():
    src = [dict(_MATCH_TOOL, parameters=dict(_MATCH_TOOL["parameters"]))]
    a = _normalize_tools(src)
    b = _normalize_tools(src)

    a[0]["parameters"]["properties"]["injected"] = {"type": "string"}

    assert "injected" not in b[0]["parameters"]["properties"]
    assert "injected" not in src[0]["parameters"]["properties"]


def test_normalize_tools_wraps_without_mutating_source():
    src = [{"name": "f", "description": "d", "parameters": {"a": {"type": "string"}}}]
    out = _normalize_tools(src)

    out[0]["parameters"]["properties"]["b"] = {"type": "string"}

    assert "b" not in src[0]["parameters"]


def test_param_rename_does_not_leak_across_samples():
    shared = dict(_MATCH_TOOL, parameters=dict(_MATCH_TOOL["parameters"]))
    a = _task_for(shared, "a", "getMatchInfo", {"match_id": "X"})
    b = _task_for(shared, "b", "getMatchInfo", {"match_id": "X"})

    assert prepare_data._augment_param_rename(a, random.Random(0)) == "param_rename"
    new_key = a["metadata"]["augment_detail"]["new"]

    # The augmented sample stays self-consistent: schema, both tool copies and
    # label follow the rename together.
    assert new_key in a["tools"][0]["parameters"]["properties"]
    assert new_key in a["metadata"]["tools"][0]["parameters"]["properties"]
    assert new_key in a["metadata"]["ground_truth"][0]["arguments"]
    assert new_key in a["label"]

    # The sibling sample and the shared pool are untouched.
    assert "match_id" in b["tools"][0]["parameters"]["properties"]
    assert "match_id" in b["metadata"]["ground_truth"][0]["arguments"]
    assert "match_id" in shared["parameters"]["properties"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
