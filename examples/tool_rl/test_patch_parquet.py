"""Tests for the pre-3483165c parquet migration (``patch_parquet.py``).

Run from the repo root:

.. code-block:: bash

    python -m pytest examples/tool_rl/test_patch_parquet.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.patch_parquet import patch_frame, patch_row  # noqa: E402

_MATCH_DESC = "Retrieve information about a football match"


def _tool(name: str, description: str, props: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in props}},
    }


def _menu() -> list[dict]:
    return [
        _tool("getFootballMatchInfo", _MATCH_DESC, ["query_text"]),
        _tool("getMatchInfo", _MATCH_DESC, ["match_id"]),
        _tool("getFootballScore", "Retrieve the current score of a football match", ["match_id"]),
    ]


def test_drops_row_whose_label_key_left_the_schema():
    kept, drop, twins = patch_row(_menu(), [{"name": "getFootballMatchInfo", "arguments": {"match_id": "X"}}])
    assert drop is True
    assert twins == 1  # the twin is withheld first, then the row is dropped


def test_keeps_row_whose_label_key_matches():
    kept, drop, _ = patch_row(_menu(), [{"name": "getFootballMatchInfo", "arguments": {"query_text": "X"}}])
    assert drop is False
    assert [t["name"] for t in kept] == ["getFootballMatchInfo", "getFootballScore"]


def test_withholds_twin_of_a_desc_replace_label_tool():
    """A swapped description must not hide the sibling the original matched."""
    canonical = {
        "getFootballMatchInfo": _MATCH_DESC.lower(),
        "getMatchInfo": _MATCH_DESC.lower(),
        "getFootballScore": "retrieve the current score of a football match",
    }
    tools = _menu()
    tools[0]["description"] = "Simulate the orbit of a satellite around a planet."

    kept, drop, twins = patch_row(tools, [], canonical)

    assert drop is False and twins == 1
    assert [t["name"] for t in kept] == ["getFootballMatchInfo", "getFootballScore"]


def test_patch_frame_round_trips_through_pandas(tmp_path):
    """pandas returns nested columns as arrays — the frame path must survive it."""
    import pandas as pd

    rows = [
        {
            "data_source": "tool_rl",
            "prompt": [{"role": "user", "content": "match info"}],
            "tools": _menu(),
            "reward_model": {"style": "rule", "ground_truth": "Ground truth:\n  getFootballMatchInfo(...)"},
            "extra_info": {
                "index": 0,
                "tools": _menu(),
                "ground_truth_calls": json.dumps([{"name": "getFootballMatchInfo", "arguments": {"query_text": "X"}}]),
                "augmented": "",
            },
        },
        {  # unwinnable row: label keeps the pre-rename key
            "data_source": "tool_rl",
            "prompt": [{"role": "user", "content": "match info"}],
            "tools": _menu(),
            "reward_model": {"style": "rule", "ground_truth": "Ground truth:\n  getFootballMatchInfo(...)"},
            "extra_info": {
                "index": 1,
                "tools": _menu(),
                "ground_truth_calls": json.dumps([{"name": "getFootballMatchInfo", "arguments": {"match_id": "X"}}]),
                "augmented": "",
            },
        },
    ]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    patched, stats = patch_frame(pd.read_parquet(path), path.name)

    assert stats["rows_in"] == 2 and stats["rows_out"] == 1
    assert stats["dropped_unmatchable_label"] == 1
    assert stats["twins_removed"] == 1
    row = patched.to_dict("records")[0]
    assert [t["name"] for t in row["tools"]] == ["getFootballMatchInfo", "getFootballScore"]
    assert row["extra_info"]["index"] == 0
    assert [t["name"] for t in row["extra_info"]["tools"]] == ["getFootballMatchInfo", "getFootballScore"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
