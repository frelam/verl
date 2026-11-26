"""Outcome-verified vehicle, travel, and trading stateful tasks."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterator

from ..envs.base import build_env
from ..envs.mock_fc import TravelDomain, VehicleDomain
from ..rewards import dispatch_reward
from .schema_distractors import with_tool_distractors
from .schema_styles import apply_schema_style, mark_canonical
from .stateful_tasks import (
    TRADE_TOOLS,
    TRADE_WRITES,
    GeneratedTask,
    _c,
    _inject_fault,
    _s,
    _tool,
    _trade_goal,
    to_task,
    verify_task,
)

SOURCE = "vehicle_travel_mt"
LICENSE = "generated (repo-internal, no third-party data)"


def _enum(description: str, *values: str) -> dict[str, Any]:
    return {"type": "string", "description": description, "enum": list(values)}


def _number(description: str, minimum: float = 0.0) -> dict[str, Any]:
    return {"type": "number", "description": description, "minimum": minimum}


def _integer(description: str, minimum: int = 1) -> dict[str, Any]:
    return {"type": "integer", "description": description, "minimum": minimum}


VEHICLE_TOOLS = [
    _tool("check_drive_panel", VehicleDomain.TOOL_DESCRIPTIONS["check_drive_panel"], {}, []),
    _tool(
        "measure_speed",
        VehicleDomain.TOOL_DESCRIPTIONS["measure_speed"],
        {"unit": _enum("Requested display unit.", "km/h", "mph")},
        [],
    ),
    _tool(
        "dial_road_speed",
        VehicleDomain.TOOL_DESCRIPTIONS["dial_road_speed"],
        {"speed": _number("Absolute road speed."), "unit": _enum("Input speed unit.", "km/h", "mph")},
        ["speed"],
    ),
    _tool(
        "nudge_road_speed",
        VehicleDomain.TOOL_DESCRIPTIONS["nudge_road_speed"],
        {"delta": {"type": "number", "description": "Signed road-speed change."},
         "unit": _enum("Input delta unit.", "km/h", "mph")},
        ["delta"],
    ),
    _tool("check_gear", VehicleDomain.TOOL_DESCRIPTIONS["check_gear"], {}, []),
    _tool(
        "select_gear",
        VehicleDomain.TOOL_DESCRIPTIONS["select_gear"],
        {"gear": _enum("Transmission position.", "P", "R", "N", "D")},
        ["gear"],
    ),
    _tool("check_lights", VehicleDomain.TOOL_DESCRIPTIONS["check_lights"], {}, []),
    _tool(
        "set_light_mode",
        VehicleDomain.TOOL_DESCRIPTIONS["set_light_mode"],
        {"mode": _enum("Exterior lighting mode.", "off", "low", "high", "auto")},
        ["mode"],
    ),
    _tool("check_climate", VehicleDomain.TOOL_DESCRIPTIONS["check_climate"], {}, []),
    _tool(
        "configure_climate",
        VehicleDomain.TOOL_DESCRIPTIONS["configure_climate"],
        {
            "temperature_c": {"type": "number", "minimum": 16, "maximum": 30},
            "fan": {"type": "integer", "minimum": 0, "maximum": 7},
            "ac_on": {"type": "boolean"},
        },
        [],
    ),
]
VEHICLE_WRITES = {"select_gear", "dial_road_speed", "nudge_road_speed", "set_light_mode", "configure_climate"}

TRAVEL_TOOLS = [
    _tool(
        "scan_journeys",
        TravelDomain.TOOL_DESCRIPTIONS["scan_journeys"],
        {
            "origin": _s("Three-letter origin code."),
            "destination": _s("Three-letter destination code."),
            "departure_date": _s("ISO departure date."),
            "passengers": _integer("Number of travellers."),
        },
        ["origin", "destination", "departure_date"],
    ),
    _tool("show_journey", TravelDomain.TOOL_DESCRIPTIONS["show_journey"],
          {"flight_id": _s("Flight identifier.")}, ["flight_id"]),
    _tool(
        "scan_lodging",
        TravelDomain.TOOL_DESCRIPTIONS["scan_lodging"],
        {
            "city": _s("Destination city."),
            "check_in": _s("ISO check-in date."),
            "check_out": _s("ISO check-out date."),
            "guests": _integer("Number of guests."),
        },
        ["city", "check_in", "check_out"],
    ),
    _tool("show_lodging", TravelDomain.TOOL_DESCRIPTIONS["show_lodging"],
          {"hotel_id": _s("Hotel identifier.")}, ["hotel_id"]),
    _tool(
        "exchange_trip_money",
        TravelDomain.TOOL_DESCRIPTIONS["exchange_trip_money"],
        {
            "amount": _number("Positive amount to convert.", 0.01),
            "from_currency": _s("Source currency code."),
            "to_currency": _s("Target currency code."),
        },
        ["amount"],
    ),
    _tool(
        "hold_flight_seats",
        TravelDomain.TOOL_DESCRIPTIONS["hold_flight_seats"],
        {
            "flight_id": _s("Flight identifier."),
            "booked_by": _s("Traveller name."),
            "passengers": _integer("Number of seats."),
            "departure_date": _s("ISO departure date."),
            "currency": _s("Budget and charged currency."),
            "budget": _number("Optional maximum total price.", 0.01),
        },
        ["flight_id", "booked_by", "passengers", "departure_date"],
    ),
    _tool(
        "hold_hotel_rooms",
        TravelDomain.TOOL_DESCRIPTIONS["hold_hotel_rooms"],
        {
            "hotel_id": _s("Hotel identifier."),
            "booked_by": _s("Lead guest name."),
            "guests": _integer("Number of guests."),
            "check_in": _s("ISO check-in date."),
            "check_out": _s("ISO check-out date."),
            "currency": _s("Budget and charged currency."),
            "budget": _number("Optional maximum total price.", 0.01),
        },
        ["hotel_id", "booked_by", "guests", "check_in", "check_out"],
    ),
    _tool("review_trip_bookings", TravelDomain.TOOL_DESCRIPTIONS["review_trip_bookings"],
          {"booked_by": _s("Optional traveller filter.")}, []),
    _tool("show_trip_booking", TravelDomain.TOOL_DESCRIPTIONS["show_trip_booking"],
          {"booking_id": _s("Reservation identifier.")}, ["booking_id"]),
    _tool("release_trip_booking", TravelDomain.TOOL_DESCRIPTIONS["release_trip_booking"],
          {"booking_id": _s("Active reservation identifier.")}, ["booking_id"]),
]
TRAVEL_WRITES = {"hold_flight_seats", "hold_hotel_rooms", "release_trip_booking"}


@dataclass
class DomainTask(GeneratedTask):
    wrong_tool: str = ""
    wrong_arguments: dict[str, Any] = field(default_factory=dict)
    decoy_call: dict[str, Any] | None = None


def _reads(rng: random.Random, calls: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(calls) > 1 and rng.random() < 0.35:
        return [calls]
    return [[call] for call in calls]


def _check(path: str, expected: Any, name: str, mode: str = "eq") -> dict[str, Any]:
    return {"check": "value", "path": path, "expected": expected, "mode": mode, "name": name}


def _unchanged(state: dict[str, Any], collection: str, identifiers: list[str]) -> list[dict[str, Any]]:
    return [
        _check(f"{collection}.{identifier}", copy.deepcopy(state[collection][identifier]), f"unchanged:{identifier}")
        for identifier in identifiers
    ]


def vehicle_cruise_setup(rng: random.Random, level: int) -> DomainTask:
    unit = rng.choice(["km/h", "mph"])
    speed = float(rng.randint(35, 75) if unit == "mph" else rng.randint(55, 115))
    speed_kmh = round(speed * (1.609344 if unit == "mph" else 1.0), 6)
    initial_ac = rng.choice([True, False])
    initial_climate = {
        "temperature_c": float(rng.randint(17, 26)),
        "fan": rng.randint(0, 7),
        "ac_on": initial_ac,
    }
    target_temperature = float(rng.randint(18, 27))
    target_fan = rng.randint(1, 6)
    target_lights = rng.choice(["low", "high", "auto"])
    state = {
        "speed_kmh": 0.0,
        "gear": "P",
        "lights": rng.choice(["off", "low"]),
        "climate": initial_climate,
    }
    plan = _reads(rng, [_c("check_drive_panel"), _c("check_gear"), _c("check_climate")])
    plan += [
        [_c("select_gear", gear="D")],
        [_c("dial_road_speed", speed=speed, unit=unit)],
        [_c("set_light_mode", mode=target_lights)],
        [_c("configure_climate", temperature_c=target_temperature, fan=target_fan)],
        [_c("measure_speed", unit=unit)],
    ]
    goal = [
        _check("speed_kmh", speed_kmh, "cruise_speed", "approx"),
        _check("gear", "D", "drive_gear"),
        _check("lights", target_lights, "lights"),
        _check("climate.temperature_c", target_temperature, "temperature", "approx"),
        _check("climate.fan", target_fan, "fan"),
        _check("climate.ac_on", initial_ac, "unchanged_ac"),
    ]
    user = (
        f"Prepare the car for driving: engage D, set cruise to {speed:g} {unit}, and select {target_lights} lights. "
        f"Set the cabin to {target_temperature:g} C with fan level {target_fan}, but leave the AC state unchanged. "
        f"Read the controls before changing them and report the resulting speed in {unit} to two decimals."
    )
    wrong_temperature = target_temperature + 1 if target_temperature < 30 else target_temperature - 1
    return DomainTask(
        "vehicle", "cruise_setup", user, VEHICLE_TOOLS, state, goal, plan,
        final_answer=f"{speed:.2f}", wrong_tool="configure_climate",
        wrong_arguments={"temperature_c": wrong_temperature},
        decoy_call=_c("configure_climate", ac_on=not initial_ac),
    )


def vehicle_stop_and_secure(rng: random.Random, level: int) -> DomainTask:
    initial_speed = float(rng.randint(45, 110))
    initial_fan = rng.randint(0, 7)
    initial_ac = rng.choice([True, False])
    target_temperature = float(rng.randint(18, 27))
    target_ac = not initial_ac
    target_lights = rng.choice(["off", "auto"])
    state = {
        "speed_kmh": initial_speed,
        "gear": "D",
        "lights": rng.choice(["low", "high"]),
        "climate": {
            "temperature_c": float(rng.randint(17, 26)),
            "fan": initial_fan,
            "ac_on": initial_ac,
        },
    }
    plan = _reads(rng, [_c("check_drive_panel"), _c("measure_speed", unit="km/h"), _c("check_lights")])
    plan += [
        [_c("nudge_road_speed", delta=-initial_speed, unit="km/h")],
        [_c("select_gear", gear="P")],
        [_c("set_light_mode", mode=target_lights)],
        [_c("configure_climate", temperature_c=target_temperature, ac_on=target_ac)],
        [_c("check_drive_panel")],
    ]
    goal = [
        _check("speed_kmh", 0.0, "stopped", "approx"),
        _check("gear", "P", "parked"),
        _check("lights", target_lights, "lights"),
        _check("climate.temperature_c", target_temperature, "temperature", "approx"),
        _check("climate.ac_on", target_ac, "ac"),
        _check("climate.fan", initial_fan, "unchanged_fan"),
    ]
    user = (
        f"Park and secure the car. Read its current panel, speed, and lights first. Use one signed cruise adjustment "
        f"to slow from the observed speed to 0 km/h, then select P and {target_lights} lights. Set the cabin to "
        f"{target_temperature:g} C and turn AC {'on' if target_ac else 'off'}, leaving the fan level untouched. "
        "Report the final speed in km/h to two decimals."
    )
    return DomainTask(
        "vehicle", "stop_and_secure", user, VEHICLE_TOOLS, state, goal, plan,
        final_answer="0.00", wrong_tool="select_gear", wrong_arguments={"gear": "D"},
        decoy_call=_c("configure_climate", fan=(initial_fan + 1) % 8),
    )


def _hotel(identifier: str, rng: random.Random, city: str, nightly: float | None = None) -> dict[str, Any]:
    return {
        "city": city,
        "nightly_usd": round(rng.uniform(85, 260), 2) if nightly is None else nightly,
        "room_capacity": rng.choice([2, 2, 3, 4]),
        "rooms_available": rng.randint(3, 9),
    }


def travel_budget_flight(rng: random.Random, level: int) -> DomainTask:
    origin, destination = rng.sample(["SEA", "SFO", "LAX", "DEN", "BOS", "AUS"], 2)
    departure = date(2027, rng.randint(1, 12), rng.randint(1, 28))
    departure_date = departure.isoformat()
    target = rng.choice(["F100", "F200"])
    decoy = "F200" if target == "F100" else "F100"
    target_price = round(rng.uniform(90, 240), 2)
    decoy_price = round(target_price + rng.uniform(25, 110), 2)
    passengers = rng.randint(1, 3)
    flights = {}
    for identifier in ("F100", "F200"):
        flights[identifier] = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "price_usd": target_price if identifier == target else decoy_price,
            "seats_available": rng.randint(passengers + 1, 10),
        }
    cities = rng.sample(["San Francisco", "Los Angeles", "Denver", "Boston"], 2)
    hotels = {identifier: _hotel(identifier, rng, city) for identifier, city in zip(("H10", "H20"), cities)}
    rates = {"USD": 1.0, "EUR": round(rng.uniform(1.05, 1.2), 4), "CNY": round(rng.uniform(0.13, 0.16), 4)}
    currency = rng.choice(list(rates))
    traveller = rng.choice(["Avery Stone", "Mina Chen", "Noah Patel", "Iris Novak"])
    total_usd = round(target_price * passengers, 2)
    exact_cost = total_usd / rates[currency]
    budget = round(exact_cost + rng.uniform(8, 35), 2)
    state = {
        "flights": flights,
        "hotels": hotels,
        "exchange_rates": rates,
        "bookings": {},
        "next_booking_id": 1,
    }
    plan = [[_c("scan_journeys", origin=origin, destination=destination,
                departure_date=departure_date, passengers=passengers)]]
    plan += _reads(rng, [
        _c("show_journey", flight_id=target),
        _c("show_journey", flight_id=decoy),
        _c("exchange_trip_money", amount=budget, from_currency=currency, to_currency="USD"),
    ])
    plan += [
        [_c("hold_flight_seats", flight_id=target, booked_by=traveller, passengers=passengers,
            departure_date=departure_date, currency=currency, budget=budget)],
        [_c("review_trip_bookings", booked_by=traveller)],
    ]
    goal = [
        _check(f"flights.{target}.seats_available", flights[target]["seats_available"] - passengers, "reserved_seats"),
        _check("bookings", 1, "one_booking", "length"),
        _check("bookings.B-1.flight_id", target, "selected_flight"),
        _check("bookings.B-1.booked_by", traveller, "traveller"),
        _check("bookings.B-1.passengers", passengers, "passengers"),
        _check("bookings.B-1.total_usd", total_usd, "total_usd", "approx"),
        _check("bookings.B-1.cost", round(exact_cost, 2), "converted_cost", "approx"),
        _check("bookings.B-1.currency", currency, "currency"),
        _check("bookings.B-1.status", "confirmed", "confirmed"),
        _check("next_booking_id", 2, "next_booking"),
    ]
    goal += _unchanged(state, "flights", [decoy])
    goal += _unchanged(state, "hotels", ["H10", "H20"])
    user = (
        f"I'm {traveller}. Find flights from {origin} to {destination} on {departure_date} for {passengers} "
        f"passenger(s), inspect both options, and check my budget conversion from {currency} to USD. Book the "
        f"lower-priced flight with a total cap of {budget:.2f} {currency}. Leave every hotel and the other flight "
        "untouched, then review my reservations and report the new booking id."
    )
    check_out = (departure + timedelta(days=2)).isoformat()
    return DomainTask(
        "travel", "budget_flight", user, TRAVEL_TOOLS, state, goal, plan,
        final_answer="B-1", wrong_tool="hold_flight_seats", wrong_arguments={"flight_id": decoy},
        decoy_call=_c("hold_hotel_rooms", hotel_id="H10", booked_by="Unrequested Guest", guests=1,
                      check_in=departure_date, check_out=check_out),
    )


def travel_hotel_and_cancellation(rng: random.Random, level: int) -> DomainTask:
    origin, destination = rng.sample(["SEA", "SFO", "LAX", "DEN"], 2)
    departure = date(2027, rng.randint(1, 12), rng.randint(1, 28))
    check_in = departure + timedelta(days=1)
    nights = rng.randint(2, 4)
    check_out = check_in + timedelta(days=nights)
    cancelled_passengers = rng.randint(1, 2)
    initial_seats = rng.randint(2, 6)
    flights = {
        "F100": {
            "origin": origin, "destination": destination, "departure_date": departure.isoformat(),
            "price_usd": round(rng.uniform(110, 260), 2), "seats_available": initial_seats,
        },
        "F200": {
            "origin": origin, "destination": destination, "departure_date": departure.isoformat(),
            "price_usd": round(rng.uniform(90, 220), 2), "seats_available": rng.randint(3, 8),
        },
    }
    city = rng.choice(["San Francisco", "Los Angeles", "Denver", "Boston"])
    target = rng.choice(["H10", "H20"])
    decoy_hotel = "H20" if target == "H10" else "H10"
    target_nightly = round(rng.uniform(90, 180), 2)
    decoy_nightly = round(target_nightly + rng.uniform(20, 80), 2)
    hotels = {}
    for identifier in ("H10", "H20"):
        hotels[identifier] = {
            "city": city,
            "nightly_usd": target_nightly if identifier == target else decoy_nightly,
            "room_capacity": 2,
            "rooms_available": rng.randint(3, 8),
        }
    guests = rng.randint(1, 4)
    rooms = (guests + 1) // 2
    traveller = rng.choice(["Riley Hart", "Kai Morgan", "Sofia Reyes", "Elliot Wang"])
    previous_traveller = rng.choice(["Jordan Lee", "Casey Kim", "Taylor Fox"])
    previous_total = round(flights["F100"]["price_usd"] * cancelled_passengers, 2)
    previous_booking = {
        "kind": "flight", "flight_id": "F100", "booked_by": previous_traveller,
        "passengers": cancelled_passengers, "departure_date": departure.isoformat(),
        "total_usd": previous_total, "cost": previous_total, "currency": "USD",
        "budget": None, "status": "confirmed",
    }
    state = {
        "flights": flights,
        "hotels": hotels,
        "exchange_rates": {"USD": 1.0, "EUR": round(rng.uniform(1.05, 1.2), 4),
                           "CNY": round(rng.uniform(0.13, 0.16), 4)},
        "bookings": {"B-0": previous_booking},
        "next_booking_id": 1,
    }
    total_usd = round(target_nightly * nights * rooms, 2)
    budget = round(total_usd + rng.uniform(10, 50), 2)
    plan = [[_c("scan_lodging", city=city, check_in=check_in.isoformat(),
                check_out=check_out.isoformat(), guests=guests)]]
    plan += _reads(rng, [
        _c("show_lodging", hotel_id=target),
        _c("show_lodging", hotel_id=decoy_hotel),
        _c("show_trip_booking", booking_id="B-0"),
    ])
    plan += [
        [_c("hold_hotel_rooms", hotel_id=target, booked_by=traveller, guests=guests,
            check_in=check_in.isoformat(), check_out=check_out.isoformat(), currency="USD", budget=budget)],
        [_c("release_trip_booking", booking_id="B-0")],
        [_c("review_trip_bookings")],
    ]
    goal = [
        _check(f"hotels.{target}.rooms_available", hotels[target]["rooms_available"] - rooms, "reserved_rooms"),
        _check("flights.F100.seats_available", initial_seats + cancelled_passengers, "restored_flight_seats"),
        _check("bookings.B-0.status", "cancelled", "old_booking_cancelled"),
        _check("bookings.B-1.kind", "hotel", "new_booking_kind"),
        _check("bookings.B-1.hotel_id", target, "selected_hotel"),
        _check("bookings.B-1.booked_by", traveller, "traveller"),
        _check("bookings.B-1.guests", guests, "guests"),
        _check("bookings.B-1.rooms", rooms, "rooms"),
        _check("bookings.B-1.nights", nights, "nights"),
        _check("bookings.B-1.total_usd", total_usd, "hotel_total", "approx"),
        _check("bookings.B-1.status", "confirmed", "new_booking_confirmed"),
        _check("bookings", 2, "booking_count", "length"),
        _check("next_booking_id", 2, "next_booking"),
    ]
    goal += _unchanged(state, "hotels", [decoy_hotel])
    goal += _unchanged(state, "flights", ["F200"])
    user = (
        f"I'm {traveller}. Search {city} lodging from {check_in.isoformat()} to {check_out.isoformat()} for "
        f"{guests} guest(s), inspect both properties, and reserve the lower-priced one with a {budget:.2f} USD "
        f"total cap. Also inspect and cancel {previous_traveller}'s existing reservation B-0. Do not alter the "
        "other hotel or flight F200. Review all bookings and report only the new hotel total in USD to two decimals."
    )
    return DomainTask(
        "travel", "hotel_and_cancellation", user, TRAVEL_TOOLS, state, goal, plan,
        final_answer=f"{total_usd:.2f}", wrong_tool="release_trip_booking",
        wrong_arguments={"booking_id": "B-1"},
        decoy_call=_c("hold_flight_seats", flight_id="F200", booked_by="Unrequested Traveller", passengers=1,
                      departure_date=departure.isoformat()),
    )


def trading_rebalance(rng: random.Random, level: int) -> DomainTask:
    symbols = rng.sample(["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "ORCL", "AMD"], 3)
    source, target, keep = symbols
    prices = {source: round(rng.uniform(100, 250), 2), target: round(rng.uniform(20, 90), 2),
              keep: round(rng.uniform(30, 180), 2)}
    sell_quantity = rng.randint(8, 15)
    buy_quantity = rng.randint(2, 5)
    holdings = {
        source: sell_quantity + rng.randint(3, 10),
        target: rng.randint(0, 8),
        keep: rng.randint(3, 9),
    }
    balance = round(rng.uniform(10, 100), 2)
    final_balance = round(balance + prices[source] * sell_quantity - prices[target] * buy_quantity, 2)
    final_holdings = dict(holdings)
    final_holdings[source] -= sell_quantity
    final_holdings[target] += buy_quantity
    state = {"balance": balance, "holdings": holdings, "prices": prices, "history": []}
    plan = _reads(rng, [_c("get_balance"), _c("get_holdings")])
    plan += _reads(rng, [_c("get_price", symbol=symbol) for symbol in symbols])
    plan += [
        [_c("sell", symbol=source, quantity=sell_quantity)],
        [_c("buy", symbol=target, quantity=buy_quantity)],
        [_c("get_history")],
    ]
    goal = _trade_goal(final_holdings, final_balance)
    goal += [
        _check("history", 2, "two_transactions", "length"),
        _check("history.0.action", "sell", "sell_recorded"),
        _check("history.0.symbol", source, "sold_symbol"),
        _check("history.1.action", "buy", "buy_recorded"),
        _check("history.1.symbol", target, "bought_symbol"),
    ]
    user = (
        f"Read my cash and holdings, then inspect the prices of {source}, {target}, and {keep}. Sell exactly "
        f"{sell_quantity} {source} shares and then buy exactly {buy_quantity} {target} shares. Keep my {keep} "
        "position completely untouched. Review the history and report the final cash balance to two decimals."
    )
    return DomainTask(
        "trading", "rebalance_with_decoy", user, TRADE_TOOLS, state, goal, plan,
        final_answer=f"{final_balance:.2f}", wrong_tool="buy", wrong_arguments={"symbol": keep},
        decoy_call=_c("sell", symbol=keep, quantity=1),
    )


def trading_sale_funded_payments(rng: random.Random, level: int) -> DomainTask:
    source, keep = rng.sample(["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "ORCL", "AMD"], 2)
    payees = rng.sample(["Avery", "Blake", "Casey", "Devon", "Emery"], 3)
    recipients, decoy_payee = payees[:2], payees[2]
    amounts = [rng.randint(40, 120), rng.randint(40, 120)]
    prices = {source: round(rng.uniform(100, 250), 2), keep: round(rng.uniform(40, 180), 2)}
    sell_quantity = rng.randint(5, 10)
    holdings = {source: sell_quantity + rng.randint(2, 8), keep: rng.randint(3, 9)}
    balance = round(rng.uniform(10, 40), 2)
    final_balance = round(balance + prices[source] * sell_quantity - sum(amounts), 2)
    state = {"balance": balance, "holdings": holdings, "prices": prices, "history": []}
    plan = _reads(rng, [_c("get_balance"), _c("get_holdings")])
    plan += _reads(rng, [_c("get_price", symbol=source), _c("get_price", symbol=keep)])
    plan += [
        [_c("sell", symbol=source, quantity=sell_quantity)],
        [_c("transfer", recipient=recipients[0], amount=amounts[0])],
        [_c("transfer", recipient=recipients[1], amount=amounts[1])],
        [_c("get_balance")],
    ]
    final_holdings = dict(holdings)
    final_holdings[source] -= sell_quantity
    goal = _trade_goal(final_holdings, final_balance)
    goal += [
        _check("history", 3, "three_transactions", "length"),
        _check("history.0.action", "sell", "sale_first"),
        _check("history.1.recipient", recipients[0], "first_payee"),
        _check("history.1.amount", float(amounts[0]), "first_amount", "approx"),
        _check("history.2.recipient", recipients[1], "second_payee"),
        _check("history.2.amount", float(amounts[1]), "second_amount", "approx"),
    ]
    user = (
        f"Check my balance, holdings, and the prices of {source} and {keep}. Sell {sell_quantity} {source} shares, "
        f"then transfer ${amounts[0]} to {recipients[0]} and ${amounts[1]} to {recipients[1]}, in that order. "
        f"Do not touch {keep} or send anything to {decoy_payee}. Report the remaining cash to two decimals."
    )
    return DomainTask(
        "trading", "sale_funded_payments", user, TRADE_TOOLS, state, goal, plan,
        final_answer=f"{final_balance:.2f}", wrong_tool="transfer", wrong_arguments={"recipient": decoy_payee},
        decoy_call=_c("sell", symbol=keep, quantity=1),
    )


DOMAIN_GENERATORS: dict[str, list[tuple[Callable[[random.Random, int], DomainTask], set[str]]]] = {
    "vehicle": [(vehicle_cruise_setup, VEHICLE_WRITES), (vehicle_stop_and_secure, VEHICLE_WRITES)],
    "travel": [(travel_budget_flight, TRAVEL_WRITES), (travel_hotel_and_cancellation, TRAVEL_WRITES)],
    "trading": [(trading_rebalance, TRADE_WRITES), (trading_sale_funded_payments, TRADE_WRITES)],
}


def _final_reply(answer: str | None) -> str:
    return f"All done. <answer>{answer}</answer>" if answer is not None else "All done."


def _replay(task: dict[str, Any], groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    env = build_env(task)
    env.reset(task)
    try:
        for group in groups:
            _, done, _ = env.step({"content": "", "tool_calls": copy.deepcopy(group)})
            if done:
                break
        if not env.done:
            env.step({"content": _final_reply(task["reference"]["final_answer"]), "tool_calls": []})
        return env.trajectory()
    finally:
        env.close()


def _negative_plans(gen: DomainTask) -> list[list[list[dict[str, Any]]]]:
    wrong = copy.deepcopy(gen.plan)
    replacements = 0
    for group in wrong:
        for call in group:
            if call["name"] == gen.wrong_tool:
                call["arguments"].update(copy.deepcopy(gen.wrong_arguments))
                replacements += 1
    if not replacements or gen.decoy_call is None:
        raise ValueError("generator is missing its negative trajectory specification")
    decoy = copy.deepcopy(gen.plan) + [[copy.deepcopy(gen.decoy_call)]]
    return [wrong, decoy]


def _verify(task: dict[str, Any], gen: DomainTask) -> None:
    verify_task(task, gen.retry_groups)
    idle = _replay(task, [])
    if dispatch_reward(task, idle)["score"] != 0.0:
        raise ValueError("idle trajectory has non-zero reward")
    for groups in _negative_plans(gen):
        trajectory = _replay(task, groups)
        if (trajectory["env_state"] or {}).get("goal_satisfied") is True:
            raise ValueError("wrong write or decoy mutation satisfies the goal")
        if dispatch_reward(task, trajectory)["score"] >= 1.0 - 1e-9:
            raise ValueError("wrong write or decoy mutation receives full reward")


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    domains: list[str] | None = None,
    fault_prob: float = 0.25,
    schema_prob: float = 0.4,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generate deterministic tasks with strict round-robin domain quotas."""
    if n < 0:
        raise ValueError("n must be non-negative")
    selected = list(domains or DOMAIN_GENERATORS)
    unknown = sorted(set(selected) - set(DOMAIN_GENERATORS))
    if unknown:
        raise ValueError(f"unknown domains {unknown}; available {sorted(DOMAIN_GENERATORS)}")
    if not selected and n:
        raise ValueError("at least one domain is required")
    stats = stats if stats is not None else {}
    stats.setdefault("rejected", {})
    stats.setdefault("emitted", 0)
    emitted = 0
    emitted_by_domain = {domain: 0 for domain in selected}
    attempts_by_domain = {domain: 0 for domain in selected}
    while emitted < n:
        domain = selected[emitted % len(selected)]
        occurrence = emitted_by_domain[domain]
        factories = DOMAIN_GENERATORS[domain]
        factory, writes = factories[occurrence % len(factories)]
        level = occurrence % 3
        accepted = False
        while not accepted:
            attempts_by_domain[domain] += 1
            attempt = attempts_by_domain[domain]
            if attempt > 100:
                raise RuntimeError(f"too many rejected tasks for {domain}: {stats['rejected']}")
            rng = random.Random(f"{seed}:{SOURCE}:{domain}:{attempt}")
            gen = factory(rng, level)
            _inject_fault(rng, gen, writes, fault_prob)
            if len(gen.plan) + 1 > 10:
                stats["rejected"]["too_long"] = stats["rejected"].get("too_long", 0) + 1
                continue
            task_id = f"{SOURCE}-{domain}-{gen.template}-{seed}-{attempt}"
            task = to_task(gen, task_id=task_id, level=level, slack=3)
            task["source"] = SOURCE
            task["meta"].update({
                "scenario": gen.template,
                "user_turn_count": 1,
                "parallel_group_count": sum(len(group) > 1 for group in gen.plan),
            })
            if verify:
                try:
                    _verify(task, gen)
                except ValueError as exc:
                    reason = str(exc).split(" (")[0][:60]
                    stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                    continue
            task = with_tool_distractors(task, style_index=emitted)
            if (emitted % 5) / 5 < schema_prob:
                task = apply_schema_style(task, style_index=emitted)
            else:
                task = mark_canonical(task)
            if verify:
                try:
                    verify_task(task, gen.retry_groups)
                except ValueError as exc:
                    reason = f"styled {str(exc).split(' (')[0][:50]}"
                    stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                    continue
            accepted = True
        emitted_by_domain[domain] += 1
        emitted += 1
        stats["emitted"] += 1
        yield task


__all__ = ["DOMAIN_GENERATORS", "LICENSE", "SOURCE", "generate_tasks"]
