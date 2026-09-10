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
    abstain_mode_from_env,
    classify_abstention,
)
from examples.tool_rl.reward.tool_rl_reward import compute_score  # noqa: E402
from examples.tool_rl.reward.verifier import _check_strict_format  # noqa: E402

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


def _extra_info(tools=None, ground_truth_calls=None, **extra):
    info = {
        "tools": tools if tools is not None else [_WEATHER_TOOL],
        "ground_truth_calls": (
            ground_truth_calls if ground_truth_calls is not None else []
        ),
        "task_id": "test",
    }
    info.update(extra)
    return info


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


def test_abstain_mode_defaults_to_keyword(monkeypatch):
    # Env unset (e.g. reward worker did not inherit the launcher shell's
    # exports) must fall back to keyword shaping, not legacy off.
    monkeypatch.delenv("TOOL_RL_ABSTAIN_MODE", raising=False)
    assert abstain_mode_from_env() == "keyword"
    res = compute_score(
        "tool_rl",
        _think_text("The Eiffel Tower is 330 metres tall."),
        "",
        _extra_info(),  # no-tool label
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["tool_correctness"] == 0.0


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


def test_classify_no_tools_needed():
    # Regression: "no tools are needed" (no adjective between "no" and
    # "tools") fell through to GUESS while the near-synonym "no available
    # tool ..." scored NO_VALID_TOOLS.
    r = classify_abstention("No tools are needed for these calculations.")
    assert r is AbstentionClass.NO_VALID_TOOLS


def test_classify_no_tools_can():
    r = classify_abstention("No tools can directly compute these values.")
    assert r is AbstentionClass.NO_VALID_TOOLS


def test_classify_there_are_no_tools():
    r = classify_abstention("There are no tools that can fetch this data.")
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


def test_keyword_no_tools_needed_full_score(keyword_mode):
    # End-to-end regression for the reported score jump: this phrasing
    # previously classified as GUESS (0.4) while the near-synonym
    # "no available tool ..." scored 1.0.
    res = compute_score(
        "tool_rl",
        _think_text("No tools are needed for these calculations."),
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
# Reward — reference-conditioned guess penalty (chitchat negatives)
# ============================================================================
#
# ToolACE chitchat / general-knowledge turns carry an empty label because
# the desired behaviour is a direct answer, and the label string embeds the
# dataset's reference response ("\nReference:\n...").  When that reference
# is itself a direct answer, guessing must NOT be penalised.

_CHITCHAT_LABEL = "\nReference:\nParis is the capital of France."


def test_reference_prefers_answer_unit():
    from examples.tool_rl.reward.abstention import reference_prefers_answer

    assert reference_prefers_answer(_CHITCHAT_LABEL) is True
    # Reference that itself asks for clarification → not a direct answer.
    assert reference_prefers_answer(
        "\nReference:\nWhich city do you mean?"
    ) is False
    # No reference (designed-abstention negatives: hammer / desc_replace /
    # no_tools) → penalty kept.  An empty reference would classify as
    # GUESS, so this guard matters.
    assert reference_prefers_answer("") is False
    assert reference_prefers_answer("Ground truth:\n  get_weather()") is False
    assert reference_prefers_answer(None) is False


def test_keyword_guess_exempt_when_reference_answers(keyword_mode):
    # Model answers the chitchat question directly and correctly: classified
    # GUESS (kept for diagnostics) but exempt from the guess penalty.
    res = compute_score(
        "tool_rl",
        _think_text("Paris is the capital of France."),
        _CHITCHAT_LABEL,
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["abstention_ref_direct"] == 1.0
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_abstain_still_ok_when_reference_answers(keyword_mode):
    # Abstaining on a chitchat sample is not penalised either (no signal
    # prefers answering over abstaining — both score Dim 1 = 1.0).
    res = compute_score(
        "tool_rl",
        _think_text("I cannot answer this — none of the available tools fits."),
        _CHITCHAT_LABEL,
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.NO_VALID_TOOLS)
    assert res["abstention_ref_direct"] == 1.0
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_guess_penalised_when_reference_clarifies(keyword_mode):
    # Reference asks for clarification → the sample IS abstention-designed,
    # so a fabricated direct answer keeps the guess penalty.
    res = compute_score(
        "tool_rl",
        _think_text("The weather in Paris is 22°C and sunny."),
        "\nReference:\nWhich city would you like the weather for?",
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["abstention_ref_direct"] == 0.0
    assert res["tool_correctness"] == 0.0
    assert res["score"] == pytest.approx(0.4)


def test_keyword_spurious_call_penalised_when_reference_answers(keyword_mode):
    # The exemption only covers no-call responses: calling a tool on a
    # chitchat sample is still a spurious call.
    res = compute_score(
        "tool_rl",
        _think_call("get_weather", {"city": "Paris"}),
        _CHITCHAT_LABEL,
        _extra_info(),
    )
    assert res["abstention_class"] == int(AbstentionClass.SPURIOUS_CALL)
    assert res["abstention_ref_direct"] == 0.0
    assert res["tool_correctness"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["score"] == pytest.approx(0.2)


# ============================================================================
# Reward — answerable_direct tag (self-computable queries)
# ============================================================================
#
# Negatives whose query the model can resolve by pure computation are
# tagged answerable_direct at data-prep time (prepare_data).  On these a
# manually computed direct answer is legitimate behaviour — no guess
# penalty — while calling a tool is still spurious.


def test_keyword_guess_exempt_when_answerable_direct(keyword_mode):
    # No reference in the label; the extra_info tag alone exempts GUESS.
    res = compute_score(
        "tool_rl",
        _think_text("12 * 7 = 84."),
        "",
        _extra_info(answerable_direct=True),
    )
    assert res["abstention_class"] == int(AbstentionClass.GUESS)
    assert res["abstention_ref_direct"] == 0.0
    assert res["abstention_answerable_direct"] == 1.0
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_abstain_still_ok_when_answerable_direct(keyword_mode):
    # Abstaining is not penalised either (no signal prefers one over the
    # other — no gradient distortion).
    res = compute_score(
        "tool_rl",
        _think_text("I cannot answer this — none of the available tools fits."),
        "",
        _extra_info(answerable_direct=True),
    )
    assert res["abstention_class"] == int(AbstentionClass.NO_VALID_TOOLS)
    assert res["abstention_answerable_direct"] == 1.0
    assert res["tool_correctness"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_keyword_spurious_call_penalised_when_answerable_direct(keyword_mode):
    # The exemption only covers no-call responses: calling a tool on a
    # self-computable query is still a spurious call.
    res = compute_score(
        "tool_rl",
        _think_call("get_weather", {"city": "Paris"}),
        "",
        _extra_info(answerable_direct=True),
    )
    assert res["abstention_class"] == int(AbstentionClass.SPURIOUS_CALL)
    assert res["tool_correctness"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["score"] == pytest.approx(0.2)


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


def test_wrapped_call_in_think_not_treated_as_call(keyword_mode):
    # Drafting a WRAPPED <tool_call> inside <think> — considering a
    # candidate call and deciding against it — is reasoning, not an
    # emitted call: a correct visible abstention must score full marks
    # (same rule as bare JSON in think, see the test above).
    resp = (
        "<think>I could try <tool_call>\n<function=get_stock_quote>\n"
        "<parameter=symbol>AAPL</parameter>\n</function>\n</tool_call> "
        "but that tool is not declared.</think>\n"
        "I don't have access to real-time stock data, and none of the "
        "available tools can retrieve it."
    )
    res = compute_score("tool_rl", resp, "", _extra_info())
    assert res["abstention_class"] == int(AbstentionClass.NO_VALID_TOOLS)
    assert res["tool_correctness"] == 1.0
    assert res["tool_call_format"] == 1.0
    assert res["format_compliance"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_think_draft_then_real_call_full_credit(keyword_mode):
    # A WRONG call drafted inside <think> followed by the real emitted
    # call after think: only the emitted call counts (Dim 1 / Dim 3),
    # and the think-internal draft must not break the Dim 2 layout bonus.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        "<think>Draft: <tool_call>\n<function=get_weather>\n"
        "<parameter=city>London</parameter>\n</function>\n</tool_call> "
        "— no wait, the user asked for Paris.</think>\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["tool_correctness"] == 1.0
    assert res["format_compliance"] == 1.0
    assert res["tool_call_format"] == 1.0
    assert res["score"] == pytest.approx(1.0)


def test_bare_json_call_earns_no_format_credit(keyword_mode):
    # A bare ``{"name": …}`` call WITHOUT the <tool_call> wrapper keeps
    # Dim 1 content credit (semantically correct call) but must earn zero
    # on both format dims: Dim 2 treats it as "no calls" (tools are
    # available → 0.0) and Dim 3 cannot match it against the label.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        "<think>The user wants the weather in Paris.</think>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["tool_correctness"] == 1.0
    assert res["format_compliance"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["score"] == pytest.approx(0.6)


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
# Regression: strict think-block format must reject stray </think> closers
# ============================================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        ("plain answer, no think block", True),
        ("<think>reasoning</think>\nanswer", True),
        ("<think>reasoning</think>\n<tool_call>x</tool_call>", True),
        ("<think>unclosed reasoning", False),
        ("stray </think> closer", False),
        ("</think> a </think> b </think>", False),
        ("<think>reasoning</think> answer </think> tail", False),
        ("<think>a</think> ok </think> and </think>", False),
        ("</think> <think>reasoning</think> answer", False),
        ("<think>a</think> <think>b</think> answer", False),
        ("<think>reasoning</think>", False),
        ("<think>r</think>\n<tool_call>x</tool_call>\n", True),
        ("<think>r</think>\n<tool_call>x</tool_call>\ntrailing text", False),
        ("<tool_call>x</tool_call> trailing text", False),
        ("plain reply, no tool call at all", True),
        # Bare JSON calls: trailing content after the last one is a
        # strict violation; the call alone (or non-call JSON) is not.
        ('<think>r</think>\n{"name": "f", "arguments": {"a": 1}}', True),
        ('<think>r</think>\n{"name": "f"}\ntrailing text', False),
        ('{"name": "f"} trailing', False),
        ("reply with non-call json {\"a\": 1} inside", True),
        ('json in think <think>{"name": "f"}</think> plain reply', True),
        # A wrapped call drafted inside think is reasoning, not an
        # emitted call — visible text after think is not "trailing
        # content after the last call".
        ("<think>draft <tool_call>x</tool_call> rethink</think>\nanswer", True),
    ],
)
def test_check_strict_format(text, expected):
    assert _check_strict_format(text) is expected


def test_stray_close_think_tag_zeroes_format(keyword_mode):
    # ``<think>...</think> <tool_call>...</tool_call> </think>`` — the
    # trailing stray closer must zero ALL dims and the total reward even
    # though the tool call itself is perfectly matched (name/param
    # sub-scores stay raw as diagnostics).
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _think_call("get_weather", {"city": "Paris"}) + "\n</think>\n"
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["name_score"] == pytest.approx(1.0)
    assert res["format_compliance"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["tool_correctness"] == 0.0
    assert res["score"] == 0.0


def test_multiple_stray_closers_zero_total(keyword_mode):
    # ``</think> ... </think> ... </think>`` repeated strays — total 0.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _think_call("get_weather", {"city": "Paris"}) + "\n</think>\n</think>\n"
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["score"] == 0.0
    assert res["tool_correctness"] == 0.0


def test_trailing_text_after_tool_call_zeroes_score(keyword_mode):
    # Content after the last ``</tool_call>`` is unreachable (the harness
    # executes the calls) — malformed layout: the total must be 0 even
    # though the call itself is perfectly matched.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        _think_call("get_weather", {"city": "Paris"})
        + "\nLet me summarize the weather for you...\n"
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["name_score"] == pytest.approx(1.0)  # raw diagnostic
    assert res["format_compliance"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["tool_correctness"] == 0.0
    assert res["score"] == 0.0


def test_bare_json_call_with_trailing_text_zeroes_score(keyword_mode):
    # Same rule for an UNWRAPPED bare JSON call: trailing text after it
    # is a strict-layout violation → total 0 (a bare JSON call with no
    # trailing text keeps its 0.6 Dim-1 fallback credit — see
    # test_bare_json_call_earns_no_format_credit).
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = (
        "<think>The user wants the weather in Paris.</think>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "Let me summarize the weather for you...\n"
    )
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["name_score"] == pytest.approx(1.0)  # raw diagnostic
    assert res["tool_correctness"] == 0.0
    assert res["format_compliance"] == 0.0
    assert res["tool_call_format"] == 0.0
    assert res["score"] == 0.0


# ============================================================================
# Repetition penalty — degenerate loops outside tool calls
# ============================================================================

def _loop_think_call(phrase: str, times: int) -> str:
    """Response whose think block repeats *phrase* *times*, then one call."""
    think = (phrase + " ") * times
    return (
        f"<think>\n{think}</think>\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    )


def test_repetition_mild_repeat_tolerated(keyword_mode):
    # "i need to check" ×2 → each 4-gram appears at most twice → zero
    # repeat events (twice = normal restatement, not a loop).
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 2)
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_repeats"] == 0
    assert res["repetition_penalty"] == 0.0
    assert res["score"] == pytest.approx(1.0)


def test_repetition_think_reply_echo_tolerated(keyword_mode):
    # Think reasoning echoed verbatim in the visible reply: every shared
    # 4-gram appears exactly twice → zero repeat events → no penalty
    # (the old count-from-the-second-occurrence rule scored -0.5 here).
    resp = (
        "<think>\nThe Eiffel Tower is 330 metres tall and located in Paris.\n</think>\n"
        "The Eiffel Tower is 330 metres tall and located in Paris."
    )
    res = compute_score("tool_rl", resp, "", _extra_info())  # no-tool label
    assert res["repetition_repeats"] == 0
    assert res["repetition_penalty"] == 0.0
    # guess: Dim 1 = 0, Dim 2 = 1.0, Dim 3 = 1.0 → 0.4
    assert res["score"] == pytest.approx(0.4)


def test_repetition_moderate_repeat_tolerated(keyword_mode):
    # "i need to check" ×6 → 5 repeat events ≤ threshold (8) → no
    # penalty: the detector only fires on death loops.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 6)
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_repeats"] == 5
    assert res["repetition_penalty"] == 0.0
    assert res["score"] == pytest.approx(1.0)


def test_repetition_loop_in_think_penalized(keyword_mode):
    # "i need to check" ×8 → 13 repeat events → 5 beyond threshold → -0.5.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 8)
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_repeats"] == 13
    assert res["repetition_penalty"] == pytest.approx(0.5)
    assert res["score"] == pytest.approx(0.5)


def test_repetition_penalty_capped(keyword_mode):
    # Pure loop ×10 → penalty capped at 1.0: a perfect call scores 0.0.
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 10)
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_penalty"] == pytest.approx(1.0)
    assert res["score"] == pytest.approx(0.0)


def test_repetition_inside_tool_call_not_penalized(keyword_mode):
    # Repetitive parameter VALUES inside <tool_call> are stripped before
    # detection — tool calls legitimately share structure.
    repeated_value = ("echo " * 30).strip()
    label_calls = [{"name": "get_weather", "arguments": {"city": repeated_value}}]
    resp = _think_call("get_weather", {"city": repeated_value})
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_penalty"] == 0.0
    assert res["score"] == pytest.approx(1.0)


def test_repetition_env_config(keyword_mode, monkeypatch):
    monkeypatch.setenv("TOOL_RL_REPEAT_FREE", "2")
    monkeypatch.setenv("TOOL_RL_REPEAT_THRESHOLD", "0")
    monkeypatch.setenv("TOOL_RL_REPEAT_PER", "0.5")
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 3)  # 1 repeat event
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["repetition_penalty"] == pytest.approx(0.5)
    assert res["score"] == pytest.approx(0.5)


# ============================================================================
# Score range guard — final reward clamped to [-1, 1]
# ============================================================================

def test_reward_clamped_at_minus_one(keyword_mode):
    # No-tool label + 10 undeclared spurious calls (Dim 1 = -1.0, Dim 2
    # = 1.0, Dim 3 = 0 → weighted -0.3) + heavy think loop (-1.0): raw
    # total -1.3 must clamp to exactly -1.0; breakdown fields stay raw.
    think = "i need to check " * 10
    calls = "\n".join(
        f"<tool_call>\n<function=ghost_tool_{i}>\n"
        f"<parameter=x>\n{i}\n</parameter>\n</function>\n</tool_call>"
        for i in range(10)
    )
    resp = f"<think>\n{think}</think>\n{calls}"
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=[]),
    )
    assert res["tool_correctness"] == pytest.approx(-1.0)  # raw
    assert res["repetition_penalty"] == pytest.approx(1.0)  # raw
    assert res["score"] == pytest.approx(-1.0)  # clamped from -1.3


def test_reward_in_range_values_untouched(keyword_mode):
    # Clamping must not distort values already inside [-1, 1].
    label_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    resp = _loop_think_call("i need to check", 8)  # -0.5 repetition
    res = compute_score(
        "tool_rl", resp, "", _extra_info(ground_truth_calls=label_calls),
    )
    assert res["score"] == pytest.approx(0.5)


# ============================================================================
# Data prep — answerable_direct tagging
# ============================================================================

def _neg_task(query: str) -> dict:
    return {
        "label": "",
        "messages": [{"role": "user", "content": query}],
        "metadata": {"ground_truth": [], "task_id": "neg"},
    }


def test_self_computable_tool_regex():
    from examples.tool_rl.prepare_data import _SELF_COMPUTABLE_TOOL_RE

    for name in ("calculate_tip", "convert_currency", "morse_encode",
                 "unit_converter", "math_solver"):
        assert _SELF_COMPUTABLE_TOOL_RE.search(name), name
    for name in ("get_weather", "search_flights", "book_hotel", "query_db"):
        assert not _SELF_COMPUTABLE_TOOL_RE.search(name), name


def test_self_computable_query_regex():
    from examples.tool_rl.prepare_data import _SELF_COMPUTABLE_QUERY_RE

    for q in ("What is 12 * 7?", "Calculate 15% of 80", "3 + 5 = ?",
              "How much is 100 / 4?", "What is the square root of 144?",
              "100 - 37"):
        assert _SELF_COMPUTABLE_QUERY_RE.search(q), q
    # Data-dependent queries and bare-dash digit pairs (dates / phone
    # numbers) must NOT match.
    for q in ("What's the weather in Paris?", "Call 555-1234",
              "Meeting on 2024-01-15", "What's the capital of France?"):
        assert not _SELF_COMPUTABLE_QUERY_RE.search(q), q


def test_tag_answerable_direct_negatives():
    from examples.tool_rl.prepare_data import tag_answerable_direct_negatives

    arith = _neg_task("What is 12 * 7?")
    weather = _neg_task("What's the weather in Paris?")
    positive = {
        "label": "Ground truth:\n  get_weather()",
        "messages": [{"role": "user", "content": "What is 12 * 7?"}],
        "metadata": {
            "ground_truth": [{"name": "get_weather", "arguments": {}}],
            "task_id": "pos",
        },
    }
    n = tag_answerable_direct_negatives([arith, weather, positive])
    assert n == 1
    assert arith["metadata"]["answerable_direct"] is True
    assert "answerable_direct" not in weather["metadata"]
    assert "answerable_direct" not in positive["metadata"]


def test_desc_replace_tags_self_computable_tool():
    import random

    from examples.tool_rl.prepare_data import _augment_desc_replace

    def _pos(tool_name: str) -> dict:
        tool = {"name": tool_name, "description": "d", "parameters": {}}
        return {
            "label": f"Ground truth:\n  {tool_name}()",
            "tools": [dict(tool)],
            "messages": [{"role": "user", "content": "q"}],
            "metadata": {
                "tools": [dict(tool)],
                "ground_truth": [{"name": tool_name, "arguments": {}}],
                "task_id": "pos",
            },
        }

    calc = _pos("calculate_tip")
    assert _augment_desc_replace(calc, random.Random(0)) == "desc_replace"
    assert calc["metadata"]["answerable_direct"] is True

    weather = _pos("get_weather")
    assert _augment_desc_replace(weather, random.Random(0)) == "desc_replace"
    assert "answerable_direct" not in weather["metadata"]


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
