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
"""Calculator tool for agentic RL.

A safe arithmetic calculator that supports +, -, *, /, **, %, parentheses,
and common math functions (sqrt, sin, cos, ...). It is intentionally
restricted to a whitelist of names so the model cannot execute arbitrary code.

This maps to capability 1 (format protocol) and 2 (tool routing) in the
agentic training plan: the model must learn *when* to call the calculator
vs. doing mental math, and *how* to format the call correctly.
"""

import ast
import math
import operator
from typing import Any, Dict, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# Whitelist of binary operators that are safe to evaluate.
_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Whitelist of unary operators.
_SAFE_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelist of callable math functions.
_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node):
    """Recursively evaluate an AST node using the whitelist."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _safe_eval_node(node.operand)
        return op_func(operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func_name = node.func.id
        func = _SAFE_FUNCS.get(func_name)
        if func is None or not callable(func):
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Name):
        val = _SAFE_FUNCS.get(node.id)
        if val is None:
            raise ValueError(f"Unknown name: {node.id}")
        return val
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    """Safely evaluate a math expression string.

    Returns the numeric result, or raises ValueError on any disallowed
    construct (imports, attribute access, calls to non-whitelisted names, etc.).
    """
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_node(tree)


class CalculatorTool(BaseTool):
    """A safe calculator tool for arithmetic and basic math functions.

    The tool keeps a per-instance call count so the reward function can
    penalise excessive calculator usage (tool-cost reward) while still
    rewarding correct usage.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "call_count": 0,
            "expressions": [],
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        expression = parameters.get("expression", "")
        if not isinstance(expression, str) or not expression.strip():
            return (
                ToolResponse(text="Error: 'expression' parameter is required and must be a non-empty string."),
                -0.05,
                {"error": "empty_expression"},
            )

        self._instance_dict[instance_id]["call_count"] += 1
        self._instance_dict[instance_id]["expressions"].append(expression)

        try:
            result = safe_calculate(expression)
            if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                return (
                    ToolResponse(text=f"Error: result is not a finite number for '{expression}'."),
                    -0.02,
                    {"error": "non_finite"},
                )
            # Format result nicely: drop trailing .0 for integers
            if isinstance(result, float) and result.is_integer():
                result_str = str(int(result))
            else:
                result_str = str(result)
            return (
                ToolResponse(text=f"{expression} = {result_str}"),
                0.0,
                {"call_count": self._instance_dict[instance_id]["call_count"]},
            )
        except Exception as e:
            return (
                ToolResponse(text=f"Error: could not evaluate '{expression}'. {type(e).__name__}: {e}"),
                -0.02,
                {"error": str(e)},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
