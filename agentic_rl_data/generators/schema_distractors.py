"""Declared-only tool distractors for schema-menu diversity.

Distractors intentionally have no domain handler: calling one returns the
existing recoverable ``unknown tool`` error and cannot change the goal state.
They are never added to GT calls or required evidence.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


def _tool(name: str, description: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    properties = properties or {"query": {"type": "string", "description": "Lookup text."}}
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }


DISTRACTOR_TOOLS = [
    _tool("lookup_timezone_offset", "Return a city's UTC offset in minutes.", {
        "city": {"type": "string"}, "day": {"type": "string"},
    }),
    _tool("convert_measurement", "Convert a numeric measurement between declared units.", {
        "value": {"type": "number"},
        "source_unit": {"type": "string", "enum": ["km", "mile", "kg", "lb"]},
        "target_unit": {"type": "string", "enum": ["km", "mile", "kg", "lb"]},
    }),
    _tool("query_calendar_conflicts", "Find overlapping events in a time range.", {
        "window": {
            "type": "object",
            "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
            "required": ["start", "end"],
        }
    }),
    _tool("lookup_transit_platform", "Find the platform for a transit service."),
    _tool("inspect_package_label", "Read handling instructions from a parcel label."),
    _tool("list_document_signatures", "List signers attached to a document."),
    _tool("get_room_capacity", "Return the rated occupancy of a venue room."),
    _tool("lookup_recipe_allergens", "Report declared allergens for a recipe."),
    _tool("inspect_sensor_battery", "Read the battery estimate for a sensor."),
    _tool("list_subscription_invoices", "List invoices for a billing subscription."),
    _tool("lookup_parking_zone", "Return rules for a parking zone identifier."),
    _tool("get_warranty_status", "Check warranty coverage for a device serial."),
    _tool("lookup_course_prerequisites", "List prerequisites for a course code."),
    _tool("inspect_storage_bucket", "Describe a cloud storage bucket policy."),
    _tool("get_print_queue", "List jobs waiting on a named printer."),
    _tool("lookup_museum_opening_hours", "Return opening intervals for a museum."),
]


def with_tool_distractors(task: Mapping[str, Any], *, style_index: int) -> dict[str, Any]:
    """Target 6/8/10/20 declared tools deterministically, preserving anchors."""
    result = copy.deepcopy(dict(task))
    tools = list(result.get("tools") or [])
    target = max(len(tools), (6, 8, 10, 20)[style_index % 4])
    existing = {str(tool.get("name")) for tool in tools}
    added = 0
    for candidate in DISTRACTOR_TOOLS:
        if len(tools) >= target:
            break
        if candidate["name"] in existing:
            continue
        tools.append(copy.deepcopy(candidate))
        existing.add(candidate["name"])
        added += 1
    result["tools"] = tools
    result["meta"]["distractor_count"] = added
    result["meta"]["tool_count_target"] = target
    return result


__all__ = ["DISTRACTOR_TOOLS", "with_tool_distractors"]
