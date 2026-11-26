"""Small, deterministic schema-style overlay for the stateful generators.

The overlay changes only the public tool interface.  Domain goals and initial
state stay canonical; declared tool names, GT calls, dialogue evidence gates,
and transient faults are rewritten together.  Every fourth tool is an
unmodified anchor so a variant never consists solely of renamed interfaces.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Mapping

NAME_STYLES = ("snake_prefix", "snake_suffix", "camelCase", "dotted")
DESCRIPTION_STYLES = ("json_schema", "openapi", "minimal")

_PARAM_ALIASES = {
    "path": "file_path",
    "source": "source_path",
    "destination": "destination_path",
    "content": "text",
    "symbol": "ticker",
    "quantity": "shares",
    "amount": "value",
    "recipient": "payee",
    "email": "email_address",
    "user_id": "customer_id",
    "order_id": "id",
    "product_id": "catalog_id",
    "address": "shipping_address",
    "reason": "cancellation_reason",
    "item_ids": "items",
    "new_item_ids": "replacement_items",
}


def _camel(name: str) -> str:
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", name) if part]
    if not parts:
        return name
    return parts[0].lower() + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _public_name(canonical: str, domain: str, style: str) -> str:
    if style == "snake_prefix":
        return f"fc_{canonical}"
    if style == "snake_suffix":
        return f"{canonical}_fc"
    if style == "camelCase":
        return _camel(canonical)
    if style == "dotted":
        prefix = re.sub(r"[^0-9A-Za-z]+", "_", domain).strip("_") or "fc"
        return f"{prefix}.{canonical}"
    raise ValueError(f"unknown tool naming style {style!r}")


def _description(tool: Mapping[str, Any], public_name: str, style: str) -> str:
    original = str(tool.get("description") or "").strip()
    first_sentence = original.split(". ", 1)[0].rstrip(".") + "."
    if style == "json_schema":
        return f"{original} Arguments are a JSON object validated by the declared schema."
    if style == "openapi":
        return (
            f"operationId: {public_name}\n"
            f"summary: {first_sentence}\n"
            "requestBody: application/json; use the declared object schema."
        )
    if style == "minimal":
        return first_sentence
    raise ValueError(f"unknown description style {style!r}")


def _alias_parameters(parameters: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Rename top-level parameters; return schema, public->canonical, canonical->public."""
    original_properties = parameters.get("properties")
    properties = dict(original_properties) if isinstance(original_properties, Mapping) else {}
    public_to_canonical: dict[str, str] = {}
    canonical_to_public: dict[str, str] = {}
    new_properties: dict[str, Any] = {}
    for canonical, schema in properties.items():
        base = _PARAM_ALIASES.get(str(canonical), f"arg_{canonical}")
        public = base
        suffix = 2
        while public in new_properties:
            public = f"{base}_{suffix}"
            suffix += 1
        public_to_canonical[public] = str(canonical)
        canonical_to_public[str(canonical)] = public
        new_properties[public] = copy.deepcopy(schema)
    required = [canonical_to_public.get(str(name), str(name)) for name in parameters.get("required") or []]
    result = copy.deepcopy(dict(parameters))
    result["type"] = "object"
    result["properties"] = new_properties
    result["required"] = required
    return result, public_to_canonical, canonical_to_public


def _nested_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    inner = copy.deepcopy(dict(parameters))
    inner["type"] = "object"
    inner.setdefault("properties", {})
    inner.setdefault("required", [])
    return {
        "type": "object",
        "properties": {"payload": inner},
        "required": ["payload"] if inner["required"] else [],
    }


def _rewrite_arguments(
    arguments: Mapping[str, Any],
    mode: str,
    canonical_to_public: Mapping[str, str],
) -> dict[str, Any]:
    if mode == "alias":
        return {
            canonical_to_public.get(str(key), str(key)): copy.deepcopy(value)
            for key, value in arguments.items()
        }
    if mode == "nested":
        return {"payload": copy.deepcopy(dict(arguments))}
    return copy.deepcopy(dict(arguments))


def mark_canonical(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with explicit canonical schema metadata."""
    result = copy.deepcopy(dict(task))
    tools = result.get("tools") or []
    canonical = [str(tool["name"]) for tool in tools if isinstance(tool, Mapping) and tool.get("name")]
    config = result["env"]["config"]
    config["tool_aliases"] = {name: name for name in canonical}
    config["param_aliases"] = {}
    config["param_wrappers"] = {}
    result["meta"].update(
        {
            "canonical_tools": canonical,
            "schema_style": "canonical",
            "description_style": "original",
            "parameter_styles": {name: "canonical" for name in canonical},
            "schema_anchor_tools": list(canonical),
        }
    )
    return result


def apply_schema_style(
    task: Mapping[str, Any],
    rng: random.Random | None = None,
    *,
    style_index: int | None = None,
) -> dict[str, Any]:
    """Apply one of four naming / three description styles to a task copy.

    ``style_index`` is preferred over random choice by generators because it
    gives deterministic stratified coverage.  Non-anchor tools alternate
    between top-level aliases and a one-level ``payload`` object.
    """
    if style_index is None:
        style_index = rng.randrange(len(NAME_STYLES) * len(DESCRIPTION_STYLES)) if rng is not None else 0
    naming = NAME_STYLES[style_index % len(NAME_STYLES)]
    description = DESCRIPTION_STYLES[(style_index // len(NAME_STYLES)) % len(DESCRIPTION_STYLES)]
    result = copy.deepcopy(dict(task))
    domain = str(result["env"]["config"].get("domain") or "fc")
    original_tools = [copy.deepcopy(tool) for tool in result.get("tools") or []]

    public_tools: list[dict[str, Any]] = []
    canonical_to_name: dict[str, str] = {}
    tool_aliases: dict[str, str] = {}
    param_aliases: dict[str, dict[str, str]] = {}
    param_wrappers: dict[str, list[str]] = {}
    parameter_styles: dict[str, str] = {}
    anchors: list[str] = []
    transforms: dict[str, tuple[str, str, dict[str, str]]] = {}

    for index, tool in enumerate(original_tools):
        canonical = str(tool["name"])
        anchor = index % 4 == 0
        public = canonical if anchor else _public_name(canonical, domain, naming)
        if public in canonical_to_name.values():
            raise ValueError(f"schema style {naming!r} produced duplicate tool name {public!r}")
        canonical_to_name[canonical] = public
        tool_aliases[public] = canonical

        parameters = tool.get("parameters")
        parameters = dict(parameters) if isinstance(parameters, Mapping) else {"type": "object", "properties": {}}
        properties = parameters.get("properties")
        has_properties = isinstance(properties, Mapping) and bool(properties)
        if anchor or not has_properties:
            mode = "canonical"
        elif (index + style_index) % 3 == 0:
            mode = "nested"
        else:
            mode = "alias"

        public_to_canonical: dict[str, str] = {}
        canonical_to_public: dict[str, str] = {}
        new_tool = copy.deepcopy(tool)
        new_tool["name"] = public
        new_tool["description"] = str(tool.get("description") or "") if anchor else _description(tool, public, description)
        if mode == "alias":
            new_parameters, public_to_canonical, canonical_to_public = _alias_parameters(parameters)
            new_tool["parameters"] = new_parameters
            param_aliases[public] = public_to_canonical
        elif mode == "nested":
            new_tool["parameters"] = _nested_parameters(parameters)
            param_wrappers[public] = ["payload"]
        else:
            new_tool["parameters"] = copy.deepcopy(parameters)
        if anchor:
            anchors.append(public)
        parameter_styles[public] = mode
        transforms[canonical] = (public, mode, canonical_to_public)
        public_tools.append(new_tool)

    def rewrite_call(call: Mapping[str, Any]) -> dict[str, Any]:
        canonical = str(call.get("name"))
        if canonical not in transforms:
            raise ValueError(f"GT call uses undeclared canonical tool {canonical!r}")
        public, mode, names = transforms[canonical]
        arguments = call.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        return {
            "name": public,
            "arguments": _rewrite_arguments(arguments, mode, names),
        }

    result["tools"] = public_tools
    result["reference"]["gt_calls"] = [
        [rewrite_call(call) for call in group]
        for group in result["reference"].get("gt_calls") or []
    ]
    config = result["env"]["config"]
    config["faults"] = [
        {**copy.deepcopy(fault), "tool": canonical_to_name.get(str(fault.get("tool")), str(fault.get("tool")))}
        for fault in config.get("faults") or []
    ]
    for stage in (config.get("dialogue") or {}).get("stages") or []:
        stage["required_tools"] = [
            canonical_to_name.get(str(name), str(name)) for name in stage.get("required_tools") or []
        ]
    config["required_parallel"] = [
        [rewrite_call(call) for call in group]
        for group in config.get("required_parallel") or []
    ]
    config["tool_aliases"] = tool_aliases
    config["param_aliases"] = param_aliases
    config["param_wrappers"] = param_wrappers

    result["meta"].update(
        {
            "canonical_tools": [str(tool["name"]) for tool in original_tools],
            "schema_style": naming,
            "description_style": description,
            "parameter_styles": parameter_styles,
            "schema_anchor_tools": anchors,
        }
    )
    return result


__all__ = [
    "DESCRIPTION_STYLES",
    "NAME_STYLES",
    "apply_schema_style",
    "mark_canonical",
]
