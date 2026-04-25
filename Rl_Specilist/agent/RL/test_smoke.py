# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke test for the agentic reward function and tools.

Run this before launching a full training run to make sure:

1. The reward function parses trajectories correctly.
2. Each reward module produces sensible values.
3. The calculator tool evaluates expressions safely.
4. The answer-submit tool gives correct feedback.

Usage::

    python -m Rl_Specilist.agent.RL.test_smoke
"""

import asyncio
import json

from Rl_Specilist.agent.RL.reward.agentic_reward import compute_score
from Rl_Specilist.agent.RL.tools.answer_submit_tool import AnswerSubmitTool
from Rl_Specilist.agent.RL.tools.calculator_tool import CalculatorTool, safe_calculate
from verl.tools.schemas import OpenAIFunctionToolSchema


# ---------------------------------------------------------------------------
# Calculator tests
# ---------------------------------------------------------------------------
def test_calculator():
    print("\n" + "=" * 60)
    print("TEST: CalculatorTool")
    print("=" * 60)

    # Test safe_calculate directly
    cases = [
        ("2 + 3", 5),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("sqrt(144)", 12),
        ("2 ** 10", 1024),
        ("log10(1000)", 3),
        ("abs(-5)", 5),
        ("max(1, 2, 3)", 3),
        ("gcd(12, 8)", 4),
    ]
    for expr, expected in cases:
        result = safe_calculate(expr)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] {expr} = {result} (expected {expected})")

    # Test rejection of dangerous input
    dangerous = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "exec('print(1)')",
    ]
    for expr in dangerous:
        try:
            safe_calculate(expr)
            print(f"  [FAIL] {expr} was NOT blocked!")
        except (ValueError, SyntaxError):
            print(f"  [OK] Blocked dangerous input: {expr[:40]}...")

    # Test the tool end-to-end
    schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "test",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    })
    tool = CalculatorTool(config={}, tool_schema=schema)

    async def run_tool():
        instance_id, _ = await tool.create()
        resp, reward, metrics = await tool.execute(instance_id, {"expression": "sqrt(144) + 2**5"})
        print(f"  Tool response: {resp.text}, reward={reward}, metrics={metrics}")
        await tool.release(instance_id)

    asyncio.run(run_tool())


# ---------------------------------------------------------------------------
# Answer submit tool tests
# ---------------------------------------------------------------------------
def test_answer_submit():
    print("\n" + "=" * 60)
    print("TEST: AnswerSubmitTool")
    print("=" * 60)

    schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "test",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    })
    tool = AnswerSubmitTool(config={}, tool_schema=schema)

    async def run_tool():
        # Create with ground_truth=42
        instance_id, _ = await tool.create(ground_truth="42", task_type="gsm8k")

        # Wrong answer
        resp, r1, _ = await tool.execute(instance_id, {"answer": "41", "confidence": 0.9})
        print(f"  Wrong answer: reward={r1}, feedback={resp.text[:60]}...")

        # Correct answer (revision)
        resp, r2, _ = await tool.execute(instance_id, {"answer": "42", "confidence": 0.8})
        print(f"  Correct revision: reward={r2}, feedback={resp.text[:60]}...")

        # Spam same answer
        resp, r3, _ = await tool.execute(instance_id, {"answer": "42", "confidence": 0.8})
        print(f"  Spam: reward={r3}")

        await tool.release(instance_id)

    asyncio.run(run_tool())


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------
def test_reward():
    print("\n" + "=" * 60)
    print("TEST: agentic_reward.compute_score")
    print("=" * 60)

    # Case 1: Correct answer, high confidence, used calculator
    traj1 = (
        "<think>I need to compute 17 * 23.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17 * 23"}}\n</tool_call>\n'
        "<think>The result is 391. I'm confident.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "391", "confidence": 0.95}}\n</tool_call>\n'
    )
    r1 = compute_score("openai/gsm8k", traj1, "391", {"task_type": "gsm8k", "num_turns": 2})
    print(f"\n  Case 1 (correct + confident + calculator):")
    print(f"    score={r1['score']:.3f}  answer={r1['answer_reward']}  cal={r1['calibration_reward']:.3f}")
    print(f"    verify={r1['verify_reward']:.3f}  epistemic={r1['epistemic_reward']:.3f}  tool_cost={r1['tool_cost']:.3f}")

    # Case 2: Wrong answer, high confidence (hallucination)
    traj2 = (
        "<think>17 * 23 is about 400.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "400", "confidence": 0.9}}\n</tool_call>\n'
    )
    r2 = compute_score("openai/gsm8k", traj2, "391", {"task_type": "gsm8k", "num_turns": 1})
    print(f"\n  Case 2 (wrong + overconfident + no tool):")
    print(f"    score={r2['score']:.3f}  answer={r2['answer_reward']}  cal={r2['calibration_reward']:.3f}")
    print(f"    epistemic={r2['epistemic_reward']:.3f} (should be negative)")

    # Case 3: Wrong but honest (low confidence)
    traj3 = (
        "<think>I'm not sure about this.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "400", "confidence": 0.3}}\n</tool_call>\n'
    )
    r3 = compute_score("openai/gsm8k", traj3, "391", {"task_type": "gsm8k", "num_turns": 1})
    print(f"\n  Case 3 (wrong + honest low confidence):")
    print(f"    score={r3['score']:.3f}  epistemic={r3['epistemic_reward']:.3f} (should be positive)")

    # Case 4: Fake citation (safety violation)
    traj4 = (
        "<think>Based on the search results, the answer is 391.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "391", "confidence": 0.8}}\n</tool_call>\n'
    )
    r4 = compute_score("openai/gsm8k", traj4, "391", {"task_type": "gsm8k", "num_turns": 1})
    print(f"\n  Case 4 (fake citation - says 'search results' but never searched):")
    print(f"    score={r4['score']:.3f}  safety_penalty={r4['safety_penalty']:.3f} (should be > 0)")

    # Case 5: Revised after feedback (verify + recovery)
    traj5 = (
        "<think>17 * 23 = 390.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "390", "confidence": 0.7}}\n</tool_call>\n'
        "<think>The tool said incorrect. Let me recalculate.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17 * 23"}}\n</tool_call>\n'
        "<think>It's 391.</think>\n"
        '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "391", "confidence": 0.9}}\n</tool_call>\n'
    )
    r5 = compute_score("openai/gsm8k", traj5, "391", {"task_type": "gsm8k", "num_turns": 3})
    print(f"\n  Case 5 (revised after feedback, ended correct):")
    print(f"    score={r5['score']:.3f}  verify={r5['verify_reward']:.3f} (should have revision bonus)")

    # Summary
    print("\n" + "-" * 60)
    print("Reward comparison (higher is better):")
    print(f"  Case 1 (correct+confident+tool):     {r1['score']:+.3f}")
    print(f"  Case 2 (wrong+overconfident):         {r2['score']:+.3f}")
    print(f"  Case 3 (wrong+honest):                {r3['score']:+.3f}")
    print(f"  Case 4 (fake citation):               {r4['score']:+.3f}")
    print(f"  Case 5 (revised->correct):            {r5['score']:+.3f}")
    print("-" * 60)
    print("Expected ordering: Case1 > Case5 > Case3 > Case2 > Case4")


if __name__ == "__main__":
    test_calculator()
    test_answer_submit()
    test_reward()
    print("\nAll smoke tests completed!")
