"""Loader tests with synthetic fixtures (no network, optional deps absent).

Run from the repository root::

    python -m pytest agentic_rl_data/tests/test_loaders.py -q

These tests were *not executed in the authoring sandbox* because ``pwsh``
fails to start (0xC0000142) and no Python process can be launched there; they
are static deliverables and must be run by the user in a working shell.
"""

from __future__ import annotations

import json

import pytest

from agentic_rl_data.loaders import available_sources, load_source
from agentic_rl_data.schema import validate_task

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def _load(name, cfg):
    tasks = list(load_source(name, cfg))
    assert tasks, f"{name}: expected at least one task"
    for task in tasks:
        assert validate_task(task) == [], f"{name}: invalid task {task.get('id')}: {validate_task(task)}"
        assert set(task["meta"]) >= {"max_turns", "difficulty", "pass_rate", "license"}
        assert set(task["reference"]) >= {
            "final_answer",
            "final_answers",
            "gt_calls",
            "gt_turns",
            "tool_returns",
            "unit_tests",
            "final_state",
        }
    return tasks


# ---------------------------------------------------------------------------
# legacy single-turn sources
# ---------------------------------------------------------------------------


def test_registry_lists_sources():
    names = available_sources()
    for expected in (
        "apigen", "sealtools", "toolace", "hammer", "apibank", "search_r1", "funreason_mt",
        "interleaved_mt", "stateful_abstain", "vehicle_travel_mt", "social_ops_mt",
        "parallel_mt", "procedural_st",
    ):
        assert expected in names


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        load_source("no_such_source")


def test_apigen_loader():
    records = [
        {
            "id": 17,
            "query": "What is the weather in Paris?",
            "tools": json.dumps([WEATHER_TOOL]),
            "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        }
    ]
    tasks = _load("apigen", {"records": records})
    task = tasks[0]
    assert task["task_type"] == "st_fc"
    assert task["env"]["type"] == "none"
    assert task["reference"]["gt_calls"] == [
        [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    ]
    assert task["tools"][0]["name"] == "get_weather"


def test_apigen_loader_preserves_parallel_calls_in_one_group():
    records = [
        {
            "id": "parallel-1",
            "query": "Check Paris and London weather.",
            "tools": json.dumps([WEATHER_TOOL]),
            "answers": json.dumps([
                {"name": "get_weather", "arguments": {"city": "Paris"}},
                {"name": "get_weather", "arguments": {"city": "London"}},
            ]),
        }
    ]
    task = _load("apigen", {"records": records})[0]
    assert len(task["reference"]["gt_calls"]) == 1
    assert len(task["reference"]["gt_calls"][0]) == 2


def test_hammer_negative_and_positive():
    records = [
        {
            "id": "neg-1",
            "query": "Can you write a poem about the sea?",
            "tools": json.dumps([WEATHER_TOOL]),
            "answers": "[]",
        },
        {
            "id": "pos-1",
            "query": "Weather in Paris?",
            "tools": json.dumps([WEATHER_TOOL]),
            "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        },
    ]
    tasks = _load("hammer", {"records": records})
    types = {task["task_type"] for task in tasks}
    assert types == {"irrelevance", "st_fc"}
    negative = next(task for task in tasks if task["task_type"] == "irrelevance")
    assert negative["reference"]["gt_calls"] == []
    positive = next(task for task in tasks if task["task_type"] == "st_fc")
    assert positive["reference"]["gt_calls"][0][0]["name"] == "get_weather"


def test_toolace_loader_parses_bracket_calls():
    system = (
        "You are a tool assistant. "
        + json.dumps([WEATHER_TOOL])
    )
    records = [
        {
            "system": system,
            "conversations": [
                {"from": "user", "value": "Weather in Paris?"},
                {"from": "assistant", "value": '[get_weather(city="Paris")]'},
                {"from": "tool", "value": '{"temp": 18}'},
                {"from": "user", "value": "And tomorrow?"},
                {"from": "assistant", "value": "I cannot tell without another call."},
            ],
        }
    ]
    tasks = _load("toolace", {"records": records})
    assert len(tasks) == 2
    first = tasks[0]
    assert first["reference"]["gt_calls"] == [
        [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    ]
    assert first["tools"][0]["name"] == "get_weather"
    assert tasks[1]["reference"]["gt_calls"] == []


def test_sealtools_loader(tmp_path):
    tool_rows = [
        {
            "api_name": "getMatchInfo",
            "api_description": "Get football match info.",
            "parameters": {"team": {"type": "str", "description": "Team"}},
            "required": ["team"],
            "field": "sports",
        },
        {
            "api_name": "getTeamRank",
            "api_description": "Get team rank.",
            "parameters": {"team": {"type": "str", "description": "Team"}},
            "required": ["team"],
            "field": "sports",
        },
    ]
    tools_path = tmp_path / "tool.jsonl"
    tools_path.write_text("\n".join(json.dumps(row) for row in tool_rows), encoding="utf-8")
    train_rows = [
        {
            "id": "s-1",
            "query": "Get the match info for team Alpha.",
            "calling": [{"api": "getMatchInfo", "parameters": {"team": "Alpha"}}],
        },
        {
            "id": "nested-1",
            "query": "Get the rank for the match winner.",
            "calling": [{"api": "getTeamRank", "parameters": {"team": "API_call_1.winner"}}],
        },
    ]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(train_rows[0]), encoding="utf-8")
    tasks = _load(
        "sealtools",
        {"tools_path": str(tools_path), "train_path": str(train_path), "n_distractors": 2},
    )
    task = tasks[0]
    assert task["reference"]["gt_calls"][0][0]["arguments"] == {"team": "Alpha"}
    assert task["task_type"] == "st_fc"


def test_apibank_loader(tmp_path):
    api_dir = tmp_path / "apis"
    api_dir.mkdir()
    (api_dir / "weather.py").write_text(
        "\n".join(
            [
                "class GetWeather:",
                '    description = "Get current weather for a city."',
                '    input_parameters = {"city": {"type": "str", "description": "City name"}}',
            ]
        ),
        encoding="utf-8",
    )
    dialogues = tmp_path / "dialogues"
    dialogues.mkdir()
    (dialogues / "sample-level-1-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"role": "User", "text": "What is the weather like in Paris?"}),
                json.dumps(
                    {
                        "role": "API",
                        "api_name": "GetWeather",
                        "param_dict": {"city": "Paris"},
                        "result": {"output": "sunny"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    tasks = _load(
        "apibank",
        {"api_dir": str(api_dir), "dialogues_path": str(dialogues), "n_distractors": 0},
    )
    task = tasks[0]
    assert task["reference"]["gt_calls"] == [[{"name": "GetWeather", "arguments": {"city": "Paris"}}]]
    assert task["messages"][-1]["content"] == "What is the weather like in Paris?"
    assert task["tools"][0]["name"] == "GetWeather"


# ---------------------------------------------------------------------------
# search / multi-hop QA
# ---------------------------------------------------------------------------


def test_search_r1_loader():
    records = [
        {
            "question": "Who wrote the novel that inspired the film Blade Runner?",
            "golden_answers": ["Philip K. Dick"],
            "id": "nq-1",
        }
    ]
    tasks = _load("search_r1", {"records": records})
    task = tasks[0]
    assert task["task_type"] == "search_qa"
    assert task["env"]["type"] == "search"
    # CONTRACTS C11 canonical search config (no legacy aliases).
    config = task["env"]["config"]
    assert {"backend", "url", "top_k", "max_rounds", "retriever"} <= set(config)
    assert config["backend"] in {"e5", "inmemory", "http"}
    assert config["top_k"] == 3
    assert config["max_rounds"] == 8
    assert task["reference"]["final_answers"] == ["Philip K. Dick"]
    assert {tool["name"] for tool in task["tools"]} == {"search", "fetch"}
    assert task["reference"]["gt_calls"] == []


def test_a2search_multi_answer():
    records = [
        {
            "question": "In which city is the Eiffel Tower located?",
            "reward_model": {
                "ground_truth": [{"answer": "Paris", "aliases": ["Paris, France"]}]
            },
            "extra_info": {"id": "a2-1"},
        }
    ]
    tasks = _load("a2search", {"records": records})
    assert tasks[0]["reference"]["final_answers"] == ["Paris", "Paris, France"]
    assert tasks[0]["task_type"] == "search_qa"


def test_asearcher_expected_turns_is_propagated():
    records = [{"question": "Hard multi-hop question?", "answer": "answer", "expected_turns": 40}]
    tasks = _load("asearcher", {"records": records})
    assert tasks[0]["meta"]["expected_turns"] == 40


# ---------------------------------------------------------------------------
# multi-turn sources (replay returns)
# ---------------------------------------------------------------------------


def _mt_record(include_returns: bool):
    messages = [
        {"role": "user", "content": "Book a flight from Alpha to Beta tomorrow."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search_flights", "arguments": '{"from": "Alpha", "to": "Beta"}'}}],
        },
    ]
    if include_returns:
        messages.append({"role": "tool", "name": "search_flights", "content": '[{"id": "F1"}]'})
    messages.extend(
        [
            {"role": "assistant", "content": "I found flight F1."},
            {"role": "user", "content": "Book it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "book_flight", "arguments": {"flight_id": "F1"}}],
            },
        ]
    )
    if include_returns:
        messages.append({"role": "tool", "name": "book_flight", "content": "confirmed"})
    return {
        "id": "fr-1",
        "tools": [
            {
                "name": "search_flights",
                "description": "Search flights.",
                "parameters": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}}},
            },
            {
                "name": "book_flight",
                "description": "Book a flight.",
                "parameters": {"type": "object", "properties": {"flight_id": {"type": "string"}}},
            },
        ],
        "messages": messages,
    }


def test_funreason_mt_real_returns():
    tasks = _load("funreason_mt", {"records": [_mt_record(True)]})
    task = tasks[0]
    assert task["task_type"] == "mt_fc"
    assert task["env"]["type"] == "mock_fc"
    assert len(task["reference"]["gt_calls"]) == 2
    assert task["reference"]["tool_returns"] is not None
    assert len(task["reference"]["tool_returns"]) == len(task["reference"]["gt_calls"])
    assert task["meta"]["gt_returns_source"] == "reference"
    assert all(message["role"] in {"system", "user"} for message in task["messages"])


def test_funreason_mt_placeholder_is_explicit():
    tasks = _load("funreason_mt", {"records": [_mt_record(False)]})
    task = tasks[0]
    assert task["meta"]["gt_returns_source"] == "placeholder"
    assert task["reference"]["tool_returns"] is None
    # gt_turns carries per-turn mappings (schema v2.2 shape).
    gt_turns = task["reference"]["gt_turns"]
    assert gt_turns and all(
        {"step", "assistant", "tool_results"} <= set(turn) for turn in gt_turns
    )


def test_funreason_mt_max_turns_is_capped_and_expected_turns_kept():
    record = _mt_record(True)
    record["expected_turns"] = 40
    tasks = _load("funreason_mt", {"records": [record]})
    task = tasks[0]
    assert task["meta"]["max_turns"] <= 15
    assert task["meta"]["expected_turns"] == 40


def test_toucan_loader_uses_reference_returns():
    tasks = _load("toucan", {"records": [_mt_record(True)], "max_samples": 1})
    assert tasks[0]["meta"]["gt_returns_source"] == "reference"
    assert tasks[0]["meta"]["gt_returns_coverage"] == 1.0


def _parallel_return_record(second_return: bool):
    messages = [
        {"role": "user", "content": "Check both flights."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "search_flights", "arguments": {"from": "Alpha"}},
            {"name": "search_flights", "arguments": {"from": "Beta"}},
        ]},
        {"role": "tool", "name": "search_flights", "content": "first"},
    ]
    if second_return:
        messages.append({"role": "tool", "name": "search_flights", "content": "second"})
    messages.append({"role": "assistant", "content": "Done."})
    return {"id": "parallel-returns", "tools": _mt_record(True)["tools"], "messages": messages}


def test_replay_loader_can_require_every_parallel_call_return():
    assert list(load_source("toucan", {
        "records": [_parallel_return_record(False)], "require_complete_returns": True,
    })) == []
    tasks = _load("toucan", {
        "records": [_parallel_return_record(True)], "require_complete_returns": True,
    })
    assert tasks[0]["meta"]["gt_returns_coverage"] == 1.0
    assert len(tasks[0]["reference"]["tool_returns"][0]) == 2


# ---------------------------------------------------------------------------
# abstain / math / code_exec
# ---------------------------------------------------------------------------


def test_when2call_tool_call_and_abstain():
    records = [
        {
            "uuid": "w-1",
            "tools": [WEATHER_TOOL],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</TOOLCALL>',
                },
            ],
        },
        {
            "uuid": "w-2",
            "tools": [WEATHER_TOOL],
            "messages": [
                {"role": "user", "content": "Fetch company details."},
                {"role": "assistant", "content": "Could you specify the company domain?"},
            ],
        },
    ]
    tasks = _load("when2call", {"records": records})
    assert tasks[0]["task_type"] == "st_fc"
    assert tasks[0]["reference"]["gt_calls"][0][0]["name"] == "get_weather"
    assert tasks[1]["task_type"] == "abstain"
    assert tasks[1]["reference"]["gt_calls"] == []


def test_retool_math_loader():
    records = [
        {
            "messages": [
                {"role": "user", "content": "What is 6 * 7?"},
                {"role": "assistant", "content": "<answer>42</answer>"},
            ]
        }
    ]
    tasks = _load("retool", {"records": records})
    task = tasks[0]
    assert task["task_type"] == "math_ci"
    assert task["reference"]["final_answer"] == "42"
    assert task["env"]["type"] == "python"
    assert task["tools"][0]["name"] == "python"


def test_code_exec_loader():
    records = [
        {
            "question": "Read two integers and print their sum.",
            "input_output": json.dumps({"inputs": ["1 2", "10 20"], "outputs": ["3", "30"]}),
            "difficulty": "introductory",
        }
    ]
    tasks = _load("code_exec", {"records": records, "kind": "apps"})
    task = tasks[0]
    assert task["task_type"] == "code_exec"
    assert task["env"]["type"] == "code_tests"
    unit_tests = json.loads(task["reference"]["unit_tests"])
    assert unit_tests["kind"] == "apps"
    assert unit_tests["inputs"] == ["1 2", "10 20"]
