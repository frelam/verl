"""Tests for the pre-3483165c parquet migration (``patch_parquet.py``).

The migration renames label parameter keys to the schema's current names —
a rename in the declared schema must be mirrored in the label — using the
property *description* as the anchor back to the original parameter, and drops
only the rows whose key cannot be mapped.

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

from examples.tool_rl.patch_parquet import patch_frame, repair_row  # noqa: E402

_DESC = "The country you want a policy for"


def _tool(properties: dict) -> dict:
    return {
        "name": "getEnergyPolicy",
        "description": "Retrieve an energy policy",
        "parameters": {"type": "object", "properties": properties},
    }


def test_repairs_key_with_the_description_anchor():
    """`country` -> `input_value` because the descriptions agree."""
    tools = [_tool({"input_value": {"type": "string", "description": _DESC}})]
    anchor = {"getEnergyPolicy": {"country": _DESC}}

    repaired, renamed = repair_row(tools, [{"name": "getEnergyPolicy", "arguments": {"country": "China"}}], anchor)

    assert repaired == [{"name": "getEnergyPolicy", "arguments": {"input_value": "China"}}]
    assert renamed == 1


def test_ignores_consistent_rows():
    tools = [_tool({"country": {"type": "string", "description": _DESC}})]
    calls = [{"name": "getEnergyPolicy", "arguments": {"country": "China"}}]

    repaired, renamed = repair_row(tools, calls, {"getEnergyPolicy": {"country": _DESC}})

    assert repaired == calls and renamed == 0


def test_anchor_disambiguates_two_generic_properties():
    """Only the description tells `input_value` from `config_option`."""
    tools = [
        _tool(
            {
                "input_value": {"type": "string", "description": "The country you want a policy for"},
                "config_option": {"type": "string", "description": "The year of the policy"},
            }
        )
    ]
    anchor = {"getEnergyPolicy": {"country": "The country you want a policy for", "year": "The year of the policy"}}
    calls = [{"name": "getEnergyPolicy", "arguments": {"country": "China", "year": "2024"}}]

    repaired, renamed = repair_row(tools, calls, anchor)

    assert repaired == [{"name": "getEnergyPolicy", "arguments": {"input_value": "China", "config_option": "2024"}}]
    assert renamed == 2


def test_falls_back_to_a_single_generic_property():
    tools = [_tool({"input_value": {"type": "string", "description": _DESC}})]

    repaired, renamed = repair_row(tools, [{"name": "getEnergyPolicy", "arguments": {"country": "China"}}], {})

    assert repaired == [{"name": "getEnergyPolicy", "arguments": {"input_value": "China"}}]
    assert renamed == 1


def test_drops_unresolvable_key():
    """Two generic candidates and no anchor: refusing beats guessing."""
    tools = [
        _tool(
            {
                "input_value": {"type": "string", "description": "A"},
                "config_option": {"type": "string", "description": "B"},
            }
        )
    ]

    repaired, renamed = repair_row(tools, [{"name": "getEnergyPolicy", "arguments": {"country": "China"}}], {})

    assert repaired is None and renamed == 0


def test_patch_frame_round_trips_through_pandas(tmp_path):
    """pandas returns nested columns as arrays — the frame path must survive it."""
    import pandas as pd

    good = {"country": {"type": "string", "description": _DESC}}
    renamed = {"input_value": {"type": "string", "description": _DESC}}
    # A sibling of the label tool: identical description, must stay in the menu.
    sibling = _tool(renamed)
    sibling["name"] = "getPolicyInfo"
    label_tool = _tool(renamed)

    def row(index: int, arguments: dict, tool: dict) -> dict:
        return {
            "data_source": "tool_rl",
            "prompt": [{"role": "user", "content": "energy policy"}],
            "tools": [tool, sibling],
            "reward_model": {"style": "rule", "ground_truth": "Ground truth:\n  getEnergyPolicy({...})"},
            "extra_info": {
                "index": index,
                "tools": [tool, sibling],
                "ground_truth_calls": json.dumps([{"name": "getEnergyPolicy", "arguments": arguments}]),
                "augmented": "",
            },
        }

    rows = [
        row(0, {"country": "China"}, label_tool),  # damaged: schema renamed the key
        row(1, {"country": "China"}, _tool(good)),  # already consistent
    ]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    patched, stats = patch_frame(pd.read_parquet(path), path.name, {"getEnergyPolicy": {"country": _DESC}})

    assert stats["rows_in"] == 2 and stats["rows_out"] == 2
    assert stats["rows_repaired"] == 1 and stats["keys_renamed"] == 1 and stats["rows_dropped"] == 0

    records = patched.to_dict("records")
    assert [r["extra_info"]["index"] for r in records] == [0, 1]
    repaired_calls = json.loads(records[0]["extra_info"]["ground_truth_calls"])
    assert repaired_calls == [{"name": "getEnergyPolicy", "arguments": {"input_value": "China"}}]
    assert "input_value" in records[0]["reward_model"]["ground_truth"]
    # The sibling is still declared: the reward accepts either tool.
    assert [t["name"] for t in records[0]["tools"]] == ["getEnergyPolicy", "getPolicyInfo"]
    assert [t["name"] for t in records[0]["extra_info"]["tools"]] == ["getEnergyPolicy", "getPolicyInfo"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
