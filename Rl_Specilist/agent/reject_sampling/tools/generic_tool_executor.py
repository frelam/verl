# Copyright 2025 Individual Contributor
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
"""Generic tool executor for ToolMind dataset.

ToolMind ships with diverse tool schemas (calculator, search, code_runner, etc.).
This module provides a ``GenericToolExecutor`` that routes execution to the
appropriate handler based on tool name, falling back to a no-op echo for
unrecognised tools so rollout never blocks.

Registered handlers:
  - calculator: safe arithmetic (reuses Rl_Specilist.agent.RL.tools.calculator_tool)
  - search / web_search: returns a stub "no results" (real search requires API key)
  - code_runner / python: executes Python in a subprocess with timeout
  - submit_answer / finish: records the answer, returns acknowledgement

Unregistered tools are echoed back with their parameters so the agent can
proceed — the DeepSeek judge will penalise nonsensical trajectories.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from typing import Any, Dict, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


def _safe_calculate(expression: str) -> float:
    """Reuse the safe evaluator from CalculatorTool."""
    import ast
    import math
    import operator

    _SAFE_BINOPS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }
    _SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    _SAFE_FUNCS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
        "gcd": math.gcd, "pi": math.pi, "e": math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            f = _SAFE_BINOPS.get(type(node.op))
            if f is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return f(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            f = _SAFE_UNARYOPS.get(type(node.op))
            if f is None:
                raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
            return f(_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn is None or not callable(fn):
                raise ValueError(f"Unsupported function: {node.func.id}")
            return fn(*[_eval(a) for a in node.args])
        if isinstance(node, ast.Name):
            v = _SAFE_FUNCS.get(node.id)
            if v is None:
                raise ValueError(f"Unknown name: {node.id}")
            return v
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    tree = ast.parse(expression.strip(), mode="eval")
    return _eval(tree)


def _run_python(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess, return stdout+stderr."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
        if not output:
            output = "(no output)"
        return output[:4000]  # Truncate to avoid blowing up context
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Tool name → handler mapping
# ---------------------------------------------------------------------------

def _handle_calculator(parameters: dict) -> tuple[str, float, dict]:
    expr = parameters.get("expression", parameters.get("formula", ""))
    if not isinstance(expr, str) or not expr.strip():
        return "Error: 'expression' is required", -0.05, {"error": "empty"}
    try:
        result = _safe_calculate(expr)
        if isinstance(result, float) and result.is_integer():
            result_str = str(int(result))
        else:
            result_str = str(result)
        return f"{expr} = {result_str}", 0.0, {"result": result_str}
    except Exception as e:
        return f"Error: {e}", -0.02, {"error": str(e)}


def _handle_search(parameters: dict) -> tuple[str, float, dict]:
    query = parameters.get("query", parameters.get("q", ""))
    # Stub: real search requires API integration. The judge will evaluate
    # whether the agent's reasoning is sound despite no real search results.
    return (
        f"Search results for '{query}': (search not available in offline mode. "
        f"Reason based on your existing knowledge.)",
        0.0,
        {"query": query, "stub": True},
    )


def _handle_code_runner(parameters: dict) -> tuple[str, float, dict]:
    code = parameters.get("code", parameters.get("script", parameters.get("python_code", "")))
    if not isinstance(code, str) or not code.strip():
        return "Error: 'code' is required", -0.05, {"error": "empty"}
    output = _run_python(code)
    return output, 0.0, {"code_length": len(code)}


def _handle_submit_answer(parameters: dict) -> tuple[str, float, dict]:
    answer = parameters.get("answer", parameters.get("result", ""))
    confidence = parameters.get("confidence", 0.5)
    return (
        f"Answer submitted: {answer} (confidence={confidence}). "
        f"Feedback will be provided by the judge.",
        0.1,  # Small positive reward for attempting
        {"submitted_answer": answer, "confidence": confidence},
    )


# Registry of known tool name patterns → handler
TOOL_HANDLERS = {
    "calculator": _handle_calculator,
    "calculate": _handle_calculator,
    "math": _handle_calculator,
    "search": _handle_search,
    "web_search": _handle_search,
    "search_engine": _handle_search,
    "code_runner": _handle_code_runner,
    "python": _handle_code_runner,
    "run_code": _handle_code_runner,
    "code_executor": _handle_code_runner,
    "submit_answer": _handle_submit_answer,
    "finish": _handle_submit_answer,
    "final_answer": _handle_submit_answer,
}


def _find_handler(tool_name: str):
    """Match tool name (case-insensitive) against known patterns."""
    name_lower = tool_name.lower()
    # Exact match
    if name_lower in TOOL_HANDLERS:
        return TOOL_HANDLERS[name_lower]
    # Substring match (e.g. "calculator_tool" → calculator)
    for key, handler in TOOL_HANDLERS.items():
        if key in name_lower:
            return handler
    return None


class GenericToolExecutor(BaseTool):
    """A generic tool executor that routes to handlers by tool name.

    Used for ToolMind dataset where each sample may have different tools.
    The executor is registered under the name ``generic_executor`` and can
    be configured via ``tool_config.yaml``.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "call_count": 0,
            "calls": [],
            "submitted_answer": None,
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        self._instance_dict[instance_id]["call_count"] += 1
        self._instance_dict[instance_id]["calls"].append({
            "tool": self.name,
            "parameters": parameters,
        })

        handler = _find_handler(self.name)
        if handler is not None:
            text, reward, metrics = handler(parameters)
            if self.name in ("submit_answer", "finish", "final_answer"):
                self._instance_dict[instance_id]["submitted_answer"] = parameters.get("answer", "")
            return ToolResponse(text=text), reward, metrics

        # Fallback: echo parameters so the agent can continue
        echo = json.dumps(parameters, ensure_ascii=False, indent=2)
        return (
            ToolResponse(text=f"[{self.name}] Received parameters:\n{echo}\n(Tool executed in stub mode.)"),
            0.0,
            {"stub": True, "tool_name": self.name},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict.get(instance_id, {})
        if state.get("submitted_answer"):
            return 0.1  # Bonus for submitting an answer
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
