"""Deterministic single-turn function-calling tasks with strict JSON labels.

The generator deliberately needs no runtime tool implementation or alias layer:
the public tool name in ``task["tools"]`` is exactly the name in
``reference.gt_calls``.  Every task has one assistant turn containing one call
group, 6--20 JSON-schema tools (including distractors), and is checked with the
real ``st_fc`` reward before emission by default.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

from ..rewards import st_fc
from ..schema import validate_task

SOURCE = "procedural_st"
LICENSE = "generated (repo-internal, no third-party data)"
SCENARIOS = ("multiple", "parallel", "parallel_multiple", "parameter_precision")
SCENARIO_WEIGHTS = (0.2, 0.2, 0.2, 0.4)
NAMING_STYLES = ("snake_prefix", "snake_suffix", "camelCase", "dotted")
DESCRIPTION_STYLES = ("json_schema", "openapi", "minimal")
MIN_TOOLS = 6
MAX_TOOLS = 20

SYSTEM_PROMPT = (
    "You are a single-turn tool-use assistant. Select the declared tools that satisfy the request and "
    "emit every independent call in one assistant turn. Preserve exact strings, numeric values, array "
    "order, and nested object structure. Use only requested arguments and omit optional parameters the "
    "user did not provide. Do not invent calls to unrelated tools."
)


@dataclass(frozen=True)
class ToolSpec:
    canonical_name: str
    summary: str
    request: str
    parameters: dict[str, Any]


def _string(description: str, *examples: str) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if examples:
        schema["examples"] = list(examples)
    return schema


def _enum(description: str, *values: str) -> dict[str, Any]:
    return {"type": "string", "description": description, "enum": list(values)}


def _integer(description: str, minimum: int, maximum: int) -> dict[str, Any]:
    return {
        "type": "integer",
        "description": description,
        "minimum": minimum,
        "maximum": maximum,
    }


def _number(
    description: str, minimum: float, maximum: float, *, multiple_of: float | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "number",
        "description": description,
        "minimum": minimum,
        "maximum": maximum,
    }
    if multiple_of is not None:
        schema["multipleOf"] = multiple_of
    return schema


def _spec(
    name: str,
    summary: str,
    request: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> ToolSpec:
    return ToolSpec(
        canonical_name=name,
        summary=summary,
        request=request,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False,
        },
    )


ACTION_SPECS = (
    _spec(
        "observatory_reserve_window",
        "Reserve an observing interval for one named sky target.",
        "Reserve an observatory observing window",
        {
            "station_code": _string("Exact observing-station identifier.", "OBS-Atacama-04", "OBS-Mauna-17", "OBS-Nubia-22"),
            "target_label": _string("Case-sensitive catalogue label of the target.", "NGC 2403 / field-A", "M 81 / north+2", "IC 1613 :: core"),
            "minutes": _integer("Requested interval length in minutes.", 15, 180),
        },
    ),
    _spec(
        "ferry_reserve_deck_space",
        "Hold deck capacity for vehicles on a particular ferry crossing.",
        "Hold vehicle deck space on a ferry crossing",
        {
            "vessel_code": _string("Exact vessel identifier.", "FV-Kestrel-8", "FV-Mistral-3", "FV-Cormorant-6"),
            "crossing_id": _string("Case-sensitive crossing identifier.", "XING=07:40/North", "XING=12:15/Outer", "XING=18:05/Harbour"),
            "vehicle_count": _integer("Number of vehicle spaces to hold.", 1, 12),
        },
    ),
    _spec(
        "greenhouse_schedule_irrigation",
        "Schedule one measured irrigation cycle for a greenhouse bed.",
        "Schedule a greenhouse irrigation cycle",
        {
            "bed_code": _string("Exact greenhouse-bed identifier.", "BED:C-14/orchid", "BED:A-02/fern", "BED:F-09/herb"),
            "litres": _number("Water volume in litres.", 0.5, 80.0, multiple_of=0.25),
            "start_time": _string("Exact local start timestamp.", "2027-03-14T05:45:00+08:00", "2027-04-02T22:10:00+08:00", "2027-05-19T06:30:00+08:00"),
        },
    ),
    _spec(
        "archive_order_digitisation",
        "Create a digitisation order for one archival box.",
        "Create an archive digitisation order",
        {
            "box_code": _string("Exact archival box identifier.", "BOX-71/Q4:oversize", "BOX-18/A2:letters", "BOX-93/Z1:maps"),
            "resolution_dpi": _integer("Scanning resolution in dots per inch.", 150, 1200),
            "colour_mode": _enum("Requested colour processing mode.", "greyscale", "full_colour", "bitonal"),
        },
    ),
    _spec(
        "library_place_hold",
        "Place a branch pickup hold on a catalogue volume.",
        "Place a library volume hold",
        {
            "catalogue_code": _string("Exact catalogue record identifier.", "CAT/841.9/No.17", "CAT/509.2/No.83", "CAT/733.5/No.06"),
            "branch_code": _string("Exact pickup branch code.", "BR-West+Annex", "BR-Canal/02", "BR-Hilltop/11"),
            "pickup_day": _string("Requested pickup date in ISO format.", "2027-02-18", "2027-06-07", "2027-09-21"),
        },
    ),
    _spec(
        "workshop_book_bench",
        "Book a workshop bench for a bounded work session.",
        "Book a workshop bench session",
        {
            "bench_code": _string("Exact bench identifier.", "BENCH/CNC-04", "BENCH/WOOD-12", "BENCH/PRINT-07"),
            "session_date": _string("Session date in ISO format.", "2027-01-29", "2027-03-08", "2027-08-16"),
            "hours": _integer("Whole hours to reserve.", 1, 8),
        },
    ),
    _spec(
        "courier_register_parcel",
        "Register a parcel with an explicit mass and service tier.",
        "Register a courier parcel",
        {
            "parcel_code": _string("Case-sensitive parcel identifier.", "PCL-88/North:: fragile", "PCL-14/Depot:: glass", "PCL-52/Quay:: papers"),
            "service_tier": _enum("Delivery service tier.", "same_day", "next_day", "economy"),
            "mass_kg": _number("Parcel mass in kilograms.", 0.1, 60.0, multiple_of=0.05),
        },
    ),
    _spec(
        "festival_hold_seats",
        "Hold seats in one section of a festival performance.",
        "Hold seats for a festival performance",
        {
            "performance_code": _string("Exact performance identifier.", "FEST/Nocturne-3B", "FEST/Lantern-8A", "FEST/Chorus-11C"),
            "section": _enum("Seating section.", "front_stalls", "rear_stalls", "balcony"),
            "seat_count": _integer("Number of adjacent seats requested.", 1, 10),
        },
    ),
    _spec(
        "clinic_reserve_room",
        "Reserve a clinic room for a fixed appointment interval.",
        "Reserve a clinic appointment room",
        {
            "room_code": _string("Exact clinic-room identifier.", "RM-East/4.12", "RM-Garden/1.03", "RM-North/7.20"),
            "start_time": _string("Exact local appointment timestamp.", "2027-02-11T09:20:00+08:00", "2027-04-23T14:40:00+08:00", "2027-10-05T16:10:00+08:00"),
            "duration_min": _integer("Appointment duration in minutes.", 10, 120),
        },
    ),
    _spec(
        "laboratory_queue_assay",
        "Queue an assay run using a named method and replicate count.",
        "Queue a laboratory assay run",
        {
            "sample_code": _string("Exact sample identifier.", "SMP-A17::lysate", "SMP-C04::plasma", "SMP-F29::extract"),
            "method": _enum("Assay processing method.", "colorimetric", "fluorescence", "turbidimetric"),
            "replicate_count": _integer("Number of independent replicates.", 1, 8),
        },
    ),
    _spec(
        "harbor_assign_berth",
        "Assign a vessel to a pier at a specified arrival time.",
        "Assign a harbor berth",
        {
            "vessel_code": _string("Exact arriving-vessel identifier.", "VES-Sable/902", "VES-Tern/118", "VES-Brine/445"),
            "pier_code": _string("Exact pier identifier.", "PIER-N/07", "PIER-W/12", "PIER-S/03"),
            "arrival_time": _string("Exact local arrival timestamp.", "2027-03-01T03:50:00+08:00", "2027-07-19T11:35:00+08:00", "2027-11-08T20:25:00+08:00"),
        },
    ),
    _spec(
        "energy_create_limit",
        "Create a power limit for one meter over a named period.",
        "Create a meter power limit",
        {
            "meter_code": _string("Exact meter identifier.", "MTR:Annex-04/B", "MTR:Studio-19/C", "MTR:Depot-27/A"),
            "limit_kw": _number("Power limit in kilowatts.", 0.5, 250.0, multiple_of=0.5),
            "period": _enum("Period during which the limit applies.", "overnight", "business_hours", "weekend"),
        },
    ),
)


def _precision_spec(
    name: str,
    summary: str,
    request: str,
    *,
    units: tuple[str, str, str],
    modes: tuple[str, str, str],
) -> ToolSpec:
    record_schema = {
        "type": "object",
        "description": "One independently measured line in the submission.",
        "properties": {
            "record_id": _string("Exact line identifier; punctuation is significant.", "LINE/A-01::primary", "LINE/C-17::control", "LINE/F-08::reserve"),
            "quantity": _integer("Whole-unit quantity for this line.", 1, 40),
            "disposition": _enum("Line disposition chosen by the requester.", "retain", "review", "discard"),
        },
        "required": ["record_id", "quantity", "disposition"],
        "additionalProperties": False,
    }
    payload_schema = {
        "type": "object",
        "description": "Nested submission payload; do not flatten its fields.",
        "properties": {
            "reference": _string("Exact mixed-case reference string.", "Lot A-17 / North, Row 03", "Batch Q+08 :: Cellar-2", "Set R=42 / Annex, Bay 7"),
            "mode": _enum("Submission processing mode.", *modes),
            "amount": _number("Measured amount, expressed in payload.unit.", 0.25, 300.0, multiple_of=0.25),
            "unit": _enum("Unit applying to payload.amount.", *units),
            "records": {
                "type": "array",
                "description": "Ordered measurement lines; each line is an object.",
                "items": record_schema,
                "minItems": 2,
                "maxItems": 3,
            },
        },
        "required": ["reference", "mode", "amount", "unit", "records"],
        "additionalProperties": False,
    }
    return _spec(
        name,
        summary,
        request,
        {
            "request_tag": _string("Exact outer request tag.", "REQ=Alpha/07::keep-case", "REQ=Delta/19::do-not-normalise", "REQ=Sigma/31::v2.1"),
            "payload": payload_schema,
            "temperature_c": _number("Requested temperature in degrees Celsius.", -20.0, 180.0, multiple_of=0.5),
            "operator_note": {
                **_string("Optional operator note.", "leave seals intact", "bench B only", "return original tags"),
                "default": "none",
            },
            "retry_limit": {
                **_integer("Optional number of retries.", 0, 5),
                "default": 0,
            },
        },
        required=["request_tag", "payload", "temperature_c"],
    )


PRECISION_SPECS = (
    _precision_spec(
        "fulfilment_register_manifest",
        "Register a structured fulfilment manifest without flattening its measurement lines.",
        "Register the exact structured fulfilment manifest",
        units=("g", "kg", "mg"),
        modes=("intake", "reconcile", "quarantine"),
    ),
    _precision_spec(
        "assay_submit_protocol",
        "Submit a nested assay protocol with ordered control and measurement records.",
        "Submit the exact structured assay protocol",
        units=("uL", "mL", "L"),
        modes=("screen", "confirm", "calibrate"),
    ),
    _precision_spec(
        "fabrication_create_batch",
        "Create a fabrication batch with exact units and an ordered list of component records.",
        "Create the exact structured fabrication batch",
        units=("mm", "cm", "m"),
        modes=("prototype", "production", "inspection"),
    ),
    _precision_spec(
        "cartography_publish_survey",
        "Publish a survey whose nested observations and measurement units must remain exact.",
        "Publish the exact structured cartography survey",
        units=("m", "km", "ft"),
        modes=("draft", "certified", "revision"),
    ),
)


_DISTRACTOR_NAMES = (
    ("parking_release_space", "Release a previously held parking space."),
    ("aquarium_log_feeding", "Record a feeding event for an aquarium tank."),
    ("bakery_cancel_bake", "Cancel a future bakery production slot."),
    ("cinema_change_screen", "Move a screening to a different cinema screen."),
    ("compost_record_turn", "Record that a compost pile was turned."),
    ("dojo_assign_mat", "Assign a practice mat for a martial-arts class."),
    ("florist_order_stems", "Order flower stems from a wholesale supplier."),
    ("gallery_adjust_lighting", "Adjust the lighting preset in one gallery."),
    ("hostel_release_bunk", "Release a held bunk in a hostel room."),
    ("kiln_schedule_firing", "Schedule a kiln firing program."),
    ("locker_change_pin", "Change the access PIN on a storage locker."),
    ("market_close_stall", "Close a market stall for a trading day."),
    ("nursery_tag_seedling", "Attach an inventory tag to a nursery seedling."),
    ("orchestra_tune_section", "Record a tuning session for an orchestra section."),
    ("pharmacy_stock_shelf", "Update the stock target for a pharmacy shelf."),
    ("quarry_mark_load", "Mark a quarry load for inspection."),
    ("roastery_log_cooling", "Record a coffee-batch cooling interval."),
    ("stable_book_farrier", "Book a farrier visit for a stable."),
    ("theatre_hold_prop", "Place a theatre prop on inventory hold."),
    ("vineyard_record_sample", "Record a vineyard grape sample."),
    ("warehouse_reassign_aisle", "Reassign a pallet to another warehouse aisle."),
    ("yoga_open_waitlist", "Open the waitlist for a yoga class."),
    ("zipline_inspect_harness", "Record a zipline harness inspection."),
    ("museum_rotate_exhibit", "Rotate an exhibit into a museum display case."),
)


def _make_distractors() -> tuple[ToolSpec, ...]:
    specs = []
    for index, (name, summary) in enumerate(_DISTRACTOR_NAMES):
        if index % 3 == 0:
            properties = {
                "record_code": _string("Identifier of the record to update.", "DECOY-104/A", "DECOY-287/C"),
                "reason": _enum("Reason category for the update.", "routine", "requested", "maintenance"),
            }
        elif index % 3 == 1:
            properties = {
                "record_code": _string("Identifier of the record to update.", "DECOY-319/B", "DECOY-462/D"),
                "count": _integer("Whole-number quantity associated with the update.", 1, 20),
            }
        else:
            properties = {
                "record_code": _string("Identifier of the record to update.", "DECOY-581/E", "DECOY-693/F"),
                "effective_date": _string("Date on which the update takes effect.", "2027-05-12", "2027-08-24"),
            }
        specs.append(_spec(name, summary, summary.rstrip("."), properties))
    return tuple(specs)


DISTRACTOR_SPECS = _make_distractors()
ALL_SPECS = ACTION_SPECS + PRECISION_SPECS + DISTRACTOR_SPECS


def _public_name(canonical_name: str, style: str) -> str:
    parts = canonical_name.split("_")
    if style == "snake_prefix":
        return f"fc_{canonical_name}"
    if style == "snake_suffix":
        return f"{canonical_name}_fc"
    if style == "camelCase":
        return parts[0] + "".join(part.capitalize() for part in parts[1:])
    if style == "dotted":
        return f"{parts[0]}.{'_'.join(parts[1:])}"
    raise ValueError(f"unknown naming style {style!r}")


def _description(spec: ToolSpec, style: str) -> str:
    if style == "minimal":
        return spec.summary
    if style == "openapi":
        namespace, *action = spec.canonical_name.split("_")
        path = f"/{namespace}/{'/'.join(action)}"
        return (
            f"POST {path} - {spec.summary} The request body is the JSON object defined by "
            "parameters; the response is a JSON result object."
        )
    if style == "json_schema":
        return (
            f"{spec.summary} Input is validated against the accompanying JSON Schema. Preserve required "
            "fields, enum values, defaults, nested objects, arrays, and exact scalar values; output is a "
            "JSON result object."
        )
    raise ValueError(f"unknown description style {style!r}")


def _public_tool(spec: ToolSpec, naming_style: str, description_style: str) -> dict[str, Any]:
    return {
        "name": _public_name(spec.canonical_name, naming_style),
        "description": _description(spec, description_style),
        "parameters": copy.deepcopy(spec.parameters),
    }


def _sample_value(schema: Mapping[str, Any], rng: random.Random, optional_policy: str) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return copy.deepcopy(rng.choice(enum))
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        return copy.deepcopy(rng.choice(examples))
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        value = {}
        for name, child in properties.items():
            if name in required:
                include = True
            elif optional_policy == "include":
                include = True
            elif optional_policy == "omit":
                include = False
            else:
                include = rng.random() < 0.5
            if include:
                value[name] = _sample_value(child, rng, optional_policy)
        return value
    if kind == "array":
        minimum = int(schema.get("minItems", 1))
        maximum = int(schema.get("maxItems", max(minimum, 3)))
        return [
            _sample_value(schema.get("items") or {}, rng, optional_policy)
            for _ in range(rng.randint(minimum, maximum))
        ]
    if kind == "integer":
        return rng.randint(int(schema.get("minimum", 0)), int(schema.get("maximum", 100)))
    if kind == "number":
        minimum = float(schema.get("minimum", 0.0))
        maximum = float(schema.get("maximum", 100.0))
        multiple = schema.get("multipleOf")
        if multiple:
            step = float(multiple)
            low = math.ceil(minimum / step)
            high = math.floor(maximum / step)
            return round(rng.randint(low, high) * step, 8)
        return round(rng.uniform(minimum, maximum), 4)
    if kind == "boolean":
        return rng.choice([True, False])
    if kind == "null":
        return None
    return "sample"


def _sample_arguments(
    spec: ToolSpec, rng: random.Random, optional_policy: str = "random"
) -> dict[str, Any]:
    value = _sample_value(spec.parameters, rng, optional_policy)
    if not isinstance(value, dict):  # pragma: no cover - all tool roots are objects
        raise TypeError("tool parameter root must be an object")
    return value


def _argument_key(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _distinct_arguments(spec: ToolSpec, count: int, rng: random.Random) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for _ in range(200):
        arguments = _sample_arguments(spec, rng)
        key = _argument_key(arguments)
        if key not in seen:
            seen.add(key)
            result.append(arguments)
            if len(result) == count:
                return result
    raise RuntimeError(f"could not sample {count} distinct arguments for {spec.canonical_name}")


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scenario_quotas(
    n: int,
    scenarios: Sequence[str] | None = None,
    scenario_weights: Sequence[float] | Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Return deterministic largest-remainder quotas for the requested strata."""
    selected = list(scenarios or SCENARIOS)
    unknown = sorted(set(selected) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; available {list(SCENARIOS)}")
    if len(set(selected)) != len(selected):
        raise ValueError("scenarios must not contain duplicates")
    if scenario_weights is None:
        defaults = dict(zip(SCENARIOS, SCENARIO_WEIGHTS))
        weights = [defaults[name] for name in selected]
    elif isinstance(scenario_weights, Mapping):
        weights = [float(scenario_weights.get(name, 0.0)) for name in selected]
    else:
        if len(scenario_weights) != len(selected):
            raise ValueError("scenario_weights must have one weight per selected scenario")
        weights = [float(value) for value in scenario_weights]
    if any(value < 0 or not math.isfinite(value) for value in weights):
        raise ValueError("scenario weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one scenario weight must be positive")
    raw = [n * value / total for value in weights]
    quotas = [math.floor(value) for value in raw]
    remainder = n - sum(quotas)
    order = sorted(range(len(selected)), key=lambda index: (-(raw[index] - quotas[index]), index))
    for index in order[:remainder]:
        quotas[index] += 1
    return dict(zip(selected, quotas))


def _scenario_payload(
    scenario: str, task_index: int, rng: random.Random
) -> tuple[list[ToolSpec], list[tuple[ToolSpec, dict[str, Any]]], str, str]:
    """Build unique tool specs, per-call specs/arguments, user text, optional policy."""
    if scenario == "multiple":
        specs = rng.sample(list(ACTION_SPECS), rng.randint(2, 4))
        pairs = [(spec, _sample_arguments(spec, rng)) for spec in specs]
        lines = [
            f"{index}. {spec.request}; use exactly this JSON argument object: {_json_arguments(arguments)}"
            for index, (spec, arguments) in enumerate(pairs, 1)
        ]
        user = "Complete every independent request below with one tool call per item:\n" + "\n".join(lines)
        return specs, pairs, user, "random"

    if scenario == "parallel":
        spec = rng.choice(ACTION_SPECS)
        arguments = _distinct_arguments(spec, rng.randint(2, 4), rng)
        pairs = [(spec, values) for values in arguments]
        lines = [
            f"{index}. {_json_arguments(values)}" for index, values in enumerate(arguments, 1)
        ]
        user = (
            f"{spec.request}. Use the same tool independently for each exact JSON argument object below; "
            "do not merge the requests or omit any item:\n" + "\n".join(lines)
        )
        return [spec], pairs, user, "random"

    if scenario == "parallel_multiple":
        repeated = rng.choice(ACTION_SPECS)
        others = rng.sample(
            [spec for spec in ACTION_SPECS if spec.canonical_name != repeated.canonical_name],
            rng.randint(1, 2),
        )
        repeated_arguments = _distinct_arguments(repeated, 2, rng)
        pairs = [(repeated, repeated_arguments[0]), (repeated, repeated_arguments[1])]
        pairs.extend((spec, _sample_arguments(spec, rng)) for spec in others)
        rng.shuffle(pairs)
        specs = [repeated] + others
        lines = [
            f"{index}. {spec.request}; use exactly this JSON argument object: {_json_arguments(arguments)}"
            for index, (spec, arguments) in enumerate(pairs, 1)
        ]
        user = (
            "Complete all of these independent requests in the same assistant turn. Repeated request types "
            "still require separate calls with their own arguments:\n" + "\n".join(lines)
        )
        return specs, pairs, user, "random"

    if scenario == "parameter_precision":
        spec = rng.choice(PRECISION_SPECS)
        optional_policy = "include" if task_index % 2 else "omit"
        arguments = _sample_arguments(spec, rng, optional_policy)
        user = (
            f"{spec.request}. Reproduce every nested level, array entry, number, unit, enum, and exact "
            "string from this JSON argument object; do not add optional fields that are absent: "
            + _json_arguments(arguments)
        )
        return [spec], [(spec, arguments)], user, optional_policy

    raise ValueError(f"unknown scenario {scenario!r}")


def _make_task(scenario: str, task_index: int, seed: int, rng: random.Random) -> dict[str, Any]:
    naming_style = NAMING_STYLES[task_index % len(NAMING_STYLES)]
    description_style = DESCRIPTION_STYLES[task_index % len(DESCRIPTION_STYLES)]
    specs, pairs, user, optional_policy = _scenario_payload(scenario, task_index, rng)
    calls = [
        {
            "name": _public_name(spec.canonical_name, naming_style),
            "arguments": copy.deepcopy(arguments),
        }
        for spec, arguments in pairs
    ]
    tool_count = rng.randint(MIN_TOOLS, MAX_TOOLS)
    used_names = {spec.canonical_name for spec in specs}
    distractor_pool = [spec for spec in ALL_SPECS if spec.canonical_name not in used_names]
    distractors = rng.sample(distractor_pool, tool_count - len(specs))
    tool_specs = list(specs) + distractors
    rng.shuffle(tool_specs)
    tools = [_public_tool(spec, naming_style, description_style) for spec in tool_specs]
    task_id = f"{SOURCE}-{scenario}-{seed}-{task_index:06d}"
    task = {
        "id": task_id,
        "source": SOURCE,
        "task_type": "st_fc",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "tools": tools,
        "env": {"type": "none", "config": {"max_turns": 1}},
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": [calls],
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": 1,
            "difficulty": float(len(calls)),
            "pass_rate": None,
            "license": LICENSE,
            "expected_turns": 1,
            "group_id": task_id,
            "scenario": scenario,
            "schema_style": description_style,
            "description_style": description_style,
            "naming_style": naming_style,
            "canonical_tools": [spec.canonical_name for spec in specs],
            "user_turn_count": 1,
            "n_calls": len(calls),
            "n_tools": len(tools),
            "n_distractors": len(distractors),
            "parallel_call_count": len(calls) if len(calls) > 1 else 0,
            "optional_policy": optional_policy,
        },
    }
    errors = validate_task(task)
    if errors:
        raise ValueError(f"generated invalid task {task_id}: {errors[:5]}")
    return task


def _trajectory(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "turns": [
            {
                "step": 1,
                "assistant": {
                    "content": "" if calls else "Done.",
                    "tool_calls": copy.deepcopy(calls),
                },
                "tool_results": [],
            }
        ],
        "final_answer": None,
        "num_turns": 1,
        "done": True,
        "env_state": None,
    }


def _mutate_first_argument(call: Mapping[str, Any], *, wrong_type: bool) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(call))
    arguments = mutated["arguments"]
    for name, value in arguments.items():
        if wrong_type:
            if isinstance(value, str):
                arguments[name] = {"not": "a string"}
            elif isinstance(value, bool):
                arguments[name] = 0
            elif isinstance(value, (int, float)):
                arguments[name] = "not-a-number"
            elif isinstance(value, (list, dict)):
                arguments[name] = "not-a-container"
            else:
                arguments[name] = "not-null"
        elif isinstance(value, str):
            arguments[name] = value + "-WRONG"
        elif isinstance(value, bool):
            arguments[name] = not value
        elif isinstance(value, (int, float)):
            arguments[name] = value + 1
        elif isinstance(value, list):
            arguments[name] = value[:-1]
        elif isinstance(value, dict):
            arguments[name] = {**value, "unexpected_nested_field": True}
        else:
            arguments[name] = 0
        return mutated
    arguments["unexpected_parameter"] = True
    return mutated


def verify_task(task: dict[str, Any]) -> dict[str, Any]:
    """Require GT=1 and missing/extra/wrong-value/wrong-type trajectories=0."""
    errors = validate_task(task)
    if errors:
        raise ValueError(f"schema: {errors[:5]}")
    groups = task["reference"]["gt_calls"]
    if task["task_type"] != "st_fc" or task["env"]["type"] != "none" or len(groups) != 1:
        raise ValueError("procedural_st requires st_fc/env none and exactly one GT turn group")
    calls = groups[0]
    if not calls:
        raise ValueError("procedural_st requires at least one GT call")

    positive = st_fc.reward(task, _trajectory(calls))
    if positive["score"] != 1.0:
        raise ValueError(f"GT reward is {positive['score']}: {positive['info']}")

    missing = calls[:-1]
    extra = calls + [copy.deepcopy(calls[0])]
    wrong_value = [_mutate_first_argument(calls[0], wrong_type=False)] + copy.deepcopy(calls[1:])
    wrong_type = [_mutate_first_argument(calls[0], wrong_type=True)] + copy.deepcopy(calls[1:])
    negatives = {
        "missing": missing,
        "extra": extra,
        "wrong_parameter": wrong_value,
        "wrong_type": wrong_type,
    }
    for label, candidate in negatives.items():
        score = st_fc.reward(task, _trajectory(candidate))["score"]
        if score != 0.0:
            raise ValueError(f"{label} trajectory unexpectedly scored {score}")
    return {"score": positive["score"], "num_turns": 1}


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    scenarios: Sequence[str] | None = None,
    scenario_weights: Sequence[float] | Mapping[str, float] | None = None,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield exactly ``n`` deterministic tasks with stratified scenario quotas."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    selected = list(scenarios or SCENARIOS)
    quotas = scenario_quotas(n, selected, scenario_weights)
    schedule = [scenario for scenario in selected for _ in range(quotas[scenario])]
    random.Random(f"{seed}:scenario-order").shuffle(schedule)
    stats = stats if stats is not None else {}
    stats.setdefault("emitted", 0)
    stats.setdefault("scenario_counts", {})
    for task_index, scenario in enumerate(schedule):
        rng = random.Random(f"{seed}:{task_index}:{scenario}")
        task = _make_task(scenario, task_index, seed, rng)
        if verify:
            verify_task(task)
        stats["emitted"] += 1
        counts = stats["scenario_counts"]
        counts[scenario] = counts.get(scenario, 0) + 1
        yield task


__all__ = [
    "ACTION_SPECS",
    "DESCRIPTION_STYLES",
    "DISTRACTOR_SPECS",
    "LICENSE",
    "MAX_TOOLS",
    "MIN_TOOLS",
    "NAMING_STYLES",
    "PRECISION_SPECS",
    "SCENARIOS",
    "SCENARIO_WEIGHTS",
    "SOURCE",
    "SYSTEM_PROMPT",
    "ToolSpec",
    "generate_tasks",
    "scenario_quotas",
    "verify_task",
]
