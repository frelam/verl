"""Tests for the tool_rl reward — abstention shaping + undeclared-penalty fix.

Run from the repo root:

.. code-block:: bash

    python -m pytest examples/tool_rl/reward/test_tool_rl_reward.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.tool_rl.reward.abstention import (  # noqa: E402
    AbstentionClass,
    classify_abstention,
)
from examples.tool_rl.reward.tool_rl_reward import compute_score  # noqa: E402

# ============================================================================
# Fixtures / helpers
# ============================================================================

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

_CALC_TOOL = {
    "name": "calculator",
    "description": "Evaluate a math expression.",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
    },
}


def _extra_info(tools=None, ground_truth_calls=None):
    return {
        "tools": tools if tools is not None else [_WEATHER_TOOL],
        "ground_truth_calls": (
            ground_truth_calls if ground_truth_calls is not None else []
        ),
        "task_id": "test",
    }


def _think_call(name: str, params: dict[str, str] | None = None) -> str:
    args = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in (params or {}).items()
    )
    return (
        "<think>\nreasoning step\n</think>\n"
        f"<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>"
    )


def _think_text(text: str) -> str:
    return f"<think>\nreasoning step\n</think>\n{text}"


@pytest.fixture
def keyword_mode(monkeypatch):
    monkeypatch.setenv("TOOL_RL_ABSTAIN_MODE", "keyword")


@pytest.fixture
def off_mode(monkeypatch):
    monkeypatch.setenv("TOOL_RL_ABSTAIN_MODE", "off")


# ============================================================================
# Abstention classifier
# ============================================================================

def test_classify_request_info():
    r = classify_abstention(
        _think_text("Could you please tell me which city you'd like the weather for?")
    )
    assert r is AbstentionClass.REQUEST_INFO


def test_classify_wh_question():
    assert classify_abstention(_think_text("Which city do you mean?")) is (
        AbstentionClass.REQUEST_INFO
    )


def test_classify_no_valid_tools():
    r = classify_abstention(
        _think_text("I'm sorry, I don't have access to real-time data.")
    )
    assert r is AbstentionClass.NO_VALID_TOOLS


def test_classify_none_of_tools():
    r = classify_abstention("None of the available tools can help with this request.")
    assert r is AbstentionClass.NO_VALID_TOOLS


def test_classify_guess():
    r = classify_abstention(_think_text("The weather in Paris is 22°C and sunny."))
    assert r is AbstentionClass.GUESS


def test_classify_courtesy_question_is_guess():
    # A fabricated answer followed by a trailing courtesy question is still a guess.
    r = classify_abstention(
        _think_text("The answer is 42. Let me know if you need anything else?")
    )
    assert r is AbstentionClass.GUESS


def test_classify_empty_reply_is_guess():
    assert classify_abstention("<think>\nonly reasoning\n</think>") is (
        AbstentionClass.GUESS
    )


# ============================================================================
# Reward — keyword mode, no-tool label
# ============================================================================

def test_keyword_clarify_full_score(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_text("Could you please tell me which city you mean?"),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] in (
        int(AbstentionClass.REQUEST_INFO),
        int(AbstentionClass.NO_VALID_TOOLS),
    )
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_no_valid_tools_full_score(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_text("I cannot answer this — none of the available tools fits."),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.NO_VALID_TOOLS)
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_guess_dim1_zero(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_text("The weather in Paris is 22°C and sunny."),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["tool_correctness"] == 0.0
    # Dim 2 (1.0, no calls + no tools expected) and Dim 3 (1.0) still pay out.
    assert res["score"] == pytest.approx(0.4)


def test_keyword_spurious_declared_call(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_call("get_weather", {"city": "Paris"}),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.SPURIOUS_CALL)
    assert res["tool_correctness"] == 0.0
    assert res["tool_call_format"] == 0.0  # Dim 3 hard zero
    # Only Dim 2 (well-formed think→call) pays out.
    assert res["score"] == pytest.approx(0.2)


def test_keyword_spurious_undeclared_call(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_call("fly_to_moon"),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.SPURIOUS_CALL)
    assert res["tool_correctness"] == pytest.approx(-0.1)  # 0 - 0.1×1
    assert res["tool_call_format"] == 0.0
    assert res["score"] == pytest.approx(0.14)


# ============================================================================
# Reward — legacy mode (off)
# ============================================================================

def test_off_mode_behaviour_unchanged(off_mode):
    guess = compute_score("tool_rl", _think_text("It is 22°C in Paris."), "", _extra_info())
    clarify = compute_score("tool_rl", _think_text("Which city do you mean?"), "", _extra_info())
    # Legacy: any no-call response on an empty label gets full Dim 1.
    assert guess["tool_correctness"] == 1.0
    assert clarify["tool_correctness"] == 1.0
    assert guess["abstention_class"] == -1


def test_off_mode_spurious_call_negative(off_mode):
    res = compute_score("tool_rl", _think_call("get_weather", {"city": "Paris"}), "", _extra_info())
    # Bug fix applies in legacy mode too: penalty stacks, no floor.
    assert res["tool_correctness"] == pytest.approx(-0.1)


# ============================================================================
# Regression — undeclared-penalty floor inversion
# ============================================================================

def test_undeclared_call_scores_below_declared_wrong_call(off_mode):
    """Before the fix, the max(0.0, ...) floor made an undeclared call (0.0)
    outscore a declared-but-wrong call (-0.1)."""
    tools = [_WEATHER_TOOL, _CALC_TOOL]
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]

    undeclared = compute_score(
        "tool_rl", _think_call("nonexistent_tool"), "",
        _extra_info(tools=tools, ground_truth_calls=label_calls),
    )
    declared_wrong = compute_score(
        "tool_rl", _think_call("calculator", {"expression": "1+1"}), "",
        _extra_info(tools=tools, ground_truth_calls=label_calls),
    )

    assert undeclared["tool_correctness"] == pytest.approx(-0.2)  # -0.1 guess -0.1 undeclared
    assert declared_wrong["tool_correctness"] == pytest.approx(-0.1)
    assert undeclared["tool_correctness"] < declared_wrong["tool_correctness"]


# ============================================================================
# Positive-label path unaffected by abstention shaping
# ============================================================================

def test_positive_label_match_untouched(keyword_mode):
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    res = compute_score(
        "tool_rl",
        _think_call("get_weather", {"city": "Paris"}),
        "",
        _extra_info(ground_truth_calls=label_calls),
    )
    assert res["abstention_class"] == -1
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


# ============================================================================
# Regression: hedged guesses must not be rewarded as abstentions
# ============================================================================

def test_classify_hedged_guess_is_guess():
    # A hedge ("I can't be sure") followed by a fabricated answer is the
    # cheapest reward hack — it must NOT count as a capability abstention.
    assert classify_abstention(
        _think_text("I can't be sure, but the answer is 42.")
    ) is AbstentionClass.GUESS
    assert classify_abstention(
        _think_text("I'm not able to help directly, but the answer is probably 42.")
    ) is AbstentionClass.GUESS


def test_classify_negation_with_capability_context():
    assert classify_abstention(
        _think_text("I'm unable to answer without access to a weather API.")
    ) is AbstentionClass.NO_VALID_TOOLS


def test_keyword_hedged_guess_scores_as_guess(keyword_mode):
    res = compute_score(
        "tool_rl",
        _think_text("I can't be sure, but the answer is 42."),
        "",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["tool_correctness"] == 0.0
    assert res["score"] == pytest.approx(0.4)


# ============================================================================
# Regression: JSON inside <think> is reasoning, not an emitted call
# ============================================================================

def test_json_in_think_not_treated_as_call(keyword_mode):
    # Discussing a JSON call inside <think> and then correctly abstaining
    # must not be punished as a spurious call.
    resp = (
        '<think>I could call {"name": "get_weather", "arguments": {"city": "Paris"}} '
        "but I have no real-time data.</think>\n"
        "I don't have access to real-time weather data."
    )
    res = compute_score("tool_rl", resp, "", _extra_info())
    assert res["abstention_class"] == int(AbstentionClass.NO_VALID_TOOLS)
    assert res["tool_correctness"] == 1.0
    assert res["tool_call_format"] == 1.0
    assert res["format_compliance"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_json_call_after_think_still_counts(keyword_mode):
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        "<think>The user wants the weather in Paris.</think>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


# ============================================================================
# Regression: inline JSON <tool_call> — key order must not matter
# ============================================================================

def test_inline_json_call_reordered_keys(keyword_mode):
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        "<think>Let me check the weather.</think>\n"
        '<tool_call>\n{"arguments": {"city": "Paris"}, "name": "get_weather"}\n</tool_call>'
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["tool_correctness"] == 1.0
    assert res["format_compliance"] == 1.0
    assert res["score"] == pytest.approx(1.0)


# ============================================================================
# Regression: conversation groups must not straddle train/val
# ============================================================================

def test_split_group_key():
    from examples.tool_rl.prepare_data import _split_group_key

    assert _split_group_key({"metadata": {"task_id": "toolace-3-t1"}}) == "toolace-3"
    assert _split_group_key({"metadata": {"task_id": "apibank-weather-c2"}}) == "apibank-weather"
    assert _split_group_key({"metadata": {"task_id": "multi_turn_sql_5-t0"}}) == "multi_turn_sql_5"
    assert _split_group_key({"metadata": {"task_id": "apigen-7"}}) == "apigen-7"


def test_group_aware_split_keeps_conversations_together():
    from examples.tool_rl.prepare_data import group_aware_split

    tasks = [
        {"metadata": {"task_id": "toolace-0-t0"}},
        {"metadata": {"task_id": "toolace-0-t1"}},
        {"metadata": {"task_id": "apibank-x-c0"}},
        {"metadata": {"task_id": "apibank-x-c1"}},
        {"metadata": {"task_id": "apigen-3"}},
    ]
    train, val = group_aware_split(tasks, n_val=2)
    val_ids = {t["metadata"]["task_id"] for t in val}
    train_ids = {t["metadata"]["task_id"] for t in train}
    for prefix in ("toolace-0", "apibank-x"):
        in_val = any(i.startswith(prefix) for i in val_ids)
        in_train = any(i.startswith(prefix) for i in train_ids)
        assert not (in_val and in_train), f"{prefix} straddles train/val"
    assert len(val) <= 2
    assert len(train) + len(val) == 5
