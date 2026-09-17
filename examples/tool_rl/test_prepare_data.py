"""Tests for tool_rl data preparation — schema isolation + twin-tool sampling.

Two regressions are covered:

1. ``_normalize_tools`` used to hand the *same* ``parameters`` object to every
   sample that declared a tool (the Seal-Tools / API-Bank loaders share one
   ``schemas`` pool).  ``_augment_param_rename`` mutates that object in place,
   so a rename performed for one sample silently rewrote every other sample's
   declared schema while leaving *their* labels untouched: 331/5000 Seal-Tools
   rows ended up with a label parameter key that no longer existed in the
   prompt, i.e. unwinnable (a schema-conformant call caps at Dim1 = 0.5).
2. Seal-Tools puts near-identical APIs in one field (``getMatchInfo`` /
   ``getFootballMatchInfo``: identical description, both take a match id).
   ``calling`` records the API the query was reverse-engineered from, not a
   uniquely correct answer, so declaring the twin makes tool selection a coin
   flip that penalises the equally valid pick.

Run from the repo root:

.. code-block:: bash

    python -m pytest examples/tool_rl/test_prepare_data.py -v
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl import prepare_data  # noqa: E402
from examples.tool_rl.prepare_data import (  # noqa: E402
    _format_gt,
    _make_meta,
    _normalize_tools,
    _tool_identities,
    _twin_tool_names,
)

# ============================================================================
# Helpers
# ============================================================================

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


# ============================================================================
# 1. Augmentation must not leak across samples
# ============================================================================


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


# ============================================================================
# 2. Indistinguishable twin tools
# ============================================================================


def _twin_schemas() -> dict[str, dict]:
    return {
        "getMatchInfo": dict(_MATCH_TOOL),
        "getFootballMatchInfo": dict(_MATCH_TOOL, name="getFootballMatchInfo"),
        "getFootballScore": dict(
            _MATCH_TOOL,
            name="getFootballScore",
            description="Retrieve the current score of a football match",
        ),
    }


_FOOTBALL_FIELDS = {
    "getMatchInfo": "Sports/Football",
    "getFootballMatchInfo": "Sports/Football",
    "getFootballScore": "Sports/Football",
}


def test_twin_tool_names_matches_same_field_and_description():
    identity, peers = _tool_identities(_twin_schemas(), _FOOTBALL_FIELDS)

    assert _twin_tool_names({"getFootballMatchInfo"}, identity=identity, peers=peers) == {"getMatchInfo"}
    assert _twin_tool_names({"getMatchInfo"}, identity=identity, peers=peers) == {"getFootballMatchInfo"}
    # A different description in the same field is a legitimate hard negative.
    assert _twin_tool_names({"getFootballScore"}, identity=identity, peers=peers) == set()


def test_twin_tool_names_requires_same_field():
    fields = dict(_FOOTBALL_FIELDS, getMatchInfo="Sports/Cricket")
    identity, peers = _tool_identities(_twin_schemas(), fields)

    assert _twin_tool_names({"getFootballMatchInfo"}, identity=identity, peers=peers) == set()


_TOOL_JSONL = "\n".join(
    json.dumps(entry)
    for entry in (
        {
            "api_name": "getFootballMatchInfo",
            "api_description": "Retrieve information about a football match",
            "field": "Sports/Football",
            "parameters": {"match_id": {"type": "str", "description": "id"}},
            "required": ["match_id"],
        },
        {
            "api_name": "getMatchInfo",
            "api_description": "Retrieve information about a football match",
            "field": "Sports/Football",
            "parameters": {"match_id": {"type": "str", "description": "id"}},
            "required": ["match_id"],
        },
        {
            "api_name": "getTeamStats",
            "api_description": "Retrieve statistics about a football team",
            "field": "Sports/Football",
            "parameters": {"team": {"type": "str", "description": "team"}},
            "required": ["team"],
        },
        {
            "api_name": "getWeather",
            "api_description": "Retrieve the weather for a city",
            "field": "Weather",
            "parameters": {"city": {"type": "str", "description": "city"}},
            "required": ["city"],
        },
        {
            "api_name": "getForecast",
            "api_description": "Retrieve the forecast for a city",
            "field": "Weather",
            "parameters": {"city": {"type": "str", "description": "city"}},
            "required": ["city"],
        },
    )
)


def _install_fake_sealtools(monkeypatch, train_rows: list[dict]) -> None:
    """Serve synthetic Seal-Tools files instead of hitting GitHub."""
    train_jsonl = "\n".join(json.dumps(row) for row in train_rows)

    def fake_fetch(repo, relpath, branch="main", timeout=120):
        if relpath.endswith("tool.jsonl"):
            return _TOOL_JSONL
        if relpath.endswith("train.jsonl"):
            return train_jsonl
        return None

    monkeypatch.setattr(prepare_data, "_fetch_raw", fake_fetch)


def test_load_sealtools_withholds_twin(monkeypatch):
    _install_fake_sealtools(
        monkeypatch,
        [
            {
                "id": "s1",
                "query": "Get the match information for match ID 'X'.",
                "calling": [
                    {
                        "api": "getFootballMatchInfo",
                        "parameters": {"match_id": "X"},
                        "responses": ["API_call_0"],
                    }
                ],
            }
        ],
    )

    tasks = prepare_data.load_sealtools(10)

    assert len(tasks) == 1
    names = [t["name"] for t in tasks[0]["tools"]]
    assert "getFootballMatchInfo" in names
    assert "getMatchInfo" not in names
    # Selection is still non-trivial: same-field hard negatives remain.
    assert "getTeamStats" in names


def test_load_sealtools_drops_mutually_ambiguous_label(monkeypatch):
    _install_fake_sealtools(
        monkeypatch,
        [
            {
                "id": "s1",
                "query": "Get the match information for match ID 'X'.",
                "calling": [
                    {"api": "getFootballMatchInfo", "parameters": {"match_id": "X"}, "responses": []},
                    {"api": "getMatchInfo", "parameters": {"match_id": "X"}, "responses": []},
                ],
            }
        ],
    )

    assert prepare_data.load_sealtools(10) == []
