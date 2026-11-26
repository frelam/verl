from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from agentic_rl_data.envs.mock_fc import (
    DOMAINS,
    BaseDomain,
    MessagingDomain,
    MicroblogDomain,
    TicketingDomain,
    TravelDomain,
    VehicleDomain,
    evaluate_goal,
)


NEW_DOMAINS = {
    "vehicle": VehicleDomain,
    "travel": TravelDomain,
    "microblog": MicroblogDomain,
    "messaging": MessagingDomain,
    "ticketing": TicketingDomain,
}


def call(domain: BaseDomain, name: str, **arguments: Any) -> Any:
    outcome = domain.call(name, arguments)
    assert outcome["ok"], outcome["content"]
    return json.loads(outcome["content"])


def assert_error(domain: BaseDomain, name: str, **arguments: Any) -> str:
    before = domain.snapshot()
    outcome = domain.call(name, arguments)
    assert outcome["ok"] is False
    assert outcome["content"].startswith("Error:")
    assert domain.snapshot() == before, "a failed call must not leave partial writes"
    return outcome["content"]


def assert_goals(domain: BaseDomain, checks: list[dict[str, Any]]) -> None:
    results = evaluate_goal(checks, domain.snapshot())
    assert all(result["ok"] for result in results), results


def exercise_for_reset(name: str, domain: BaseDomain) -> None:
    if name == "vehicle":
        call(domain, "set_light_mode", mode="high")
    elif name == "travel":
        call(
            domain,
            "hold_flight_seats",
            flight_id="F100",
            booked_by="Reset Tester",
            passengers=1,
            departure_date="2026-12-01",
        )
    elif name == "microblog":
        call(domain, "open_author_session", handle="alice", auth_key="orchid")
        call(domain, "stash_post_draft", text="reset probe")
    elif name == "messaging":
        call(domain, "enter_chat_account", username="alice", passcode="maple")
    elif name == "ticketing":
        call(
            domain,
            "file_support_ticket",
            title="Reset probe",
            details="Temporary request",
            requester="tester",
        )
    else:
        raise AssertionError(name)


@pytest.mark.parametrize("name,domain_class", sorted(NEW_DOMAINS.items()))
def test_domain_registration_and_original_tool_descriptions(
    name: str, domain_class: type[BaseDomain]
) -> None:
    assert DOMAINS[name] is domain_class
    assert set(domain_class.TOOL_DESCRIPTIONS) == set(domain_class.HANDLERS)
    assert all(description.strip() for description in domain_class.TOOL_DESCRIPTIONS.values())
    assert len(set(domain_class.HANDLERS)) == len(domain_class.HANDLERS)
    if name == "vehicle":
        assert 8 <= len(domain_class.HANDLERS) <= 15


@pytest.mark.parametrize("name", sorted(NEW_DOMAINS))
def test_domain_reset_restores_defaults_and_isolates_instances(name: str) -> None:
    domain = NEW_DOMAINS[name]()
    expected = domain.default_state()
    exercise_for_reset(name, domain)
    assert domain.snapshot() != expected

    domain.reset()
    assert domain.snapshot() == expected

    peer = NEW_DOMAINS[name]()
    exercise_for_reset(name, peer)
    assert peer.snapshot() != expected
    assert domain.snapshot() == expected
    peer.reset()
    assert peer.snapshot() == expected


def test_vehicle_complete_flow_unit_conversion_and_goals() -> None:
    domain = VehicleDomain()
    assert call(domain, "measure_speed", unit="mph")["speed"] == 0.0
    call(domain, "select_gear", gear="D")
    call(domain, "dial_road_speed", speed=50, unit="mph")
    adjusted = call(domain, "nudge_road_speed", delta=-10, unit="mph")
    assert adjusted["speed"] == 40.0
    assert adjusted["unit"] == "mph"
    call(domain, "set_light_mode", mode="auto")
    climate = call(
        domain, "configure_climate", temperature_c=20.5, fan=4, ac_on=True
    )
    assert climate == {"temperature_c": 20.5, "fan": 4, "ac_on": True}
    panel = call(domain, "check_drive_panel")
    assert panel["speed_kmh"] == 64.37
    assert panel["speed_mph"] == 40.0
    assert call(domain, "check_gear")["gear"] == "D"
    assert call(domain, "check_lights")["lights"] == "auto"
    assert call(domain, "check_climate")["ac_on"] is True

    assert_goals(
        domain,
        [
            {"check": "value", "path": "speed_kmh", "expected": 64.37376, "mode": "approx"},
            {"check": "value", "path": "gear", "expected": "D"},
            {"check": "value", "path": "lights", "expected": "auto"},
            {"check": "value", "path": "climate.temperature_c", "expected": 20.5},
            {"check": "exists", "path": "climate.ac_on"},
        ],
    )


def test_vehicle_rejects_moving_gear_change_without_mutation() -> None:
    domain = VehicleDomain()
    message = assert_error(domain, "dial_road_speed", speed=20, unit="km/h")
    assert "gear D" in message
    call(domain, "select_gear", gear="D")
    call(domain, "dial_road_speed", speed=60)
    message = assert_error(domain, "select_gear", gear="R")
    assert "stop" in message
    assert_error(domain, "configure_climate", temperature_c=20, fan=99)


def test_travel_complete_flow_fx_budget_cancellation_and_goals() -> None:
    domain = TravelDomain()
    decoy_flight = copy.deepcopy(domain.state["flights"]["F200"])
    decoy_hotel = copy.deepcopy(domain.state["hotels"]["H20"])
    journeys = call(
        domain,
        "scan_journeys",
        origin="SEA",
        destination="SFO",
        departure_date="2026-12-01",
        passengers=2,
    )
    assert [item["flight_id"] for item in journeys] == ["F100"]
    assert call(domain, "show_journey", flight_id="F100")["seats_available"] == 8
    lodging = call(
        domain,
        "scan_lodging",
        city="San Francisco",
        check_in="2026-12-01",
        check_out="2026-12-03",
        guests=3,
    )
    assert lodging[0]["rooms_needed"] == 2
    assert lodging[0]["nights"] == 2
    assert call(domain, "show_lodging", hotel_id="H10")["rooms_available"] == 3
    fx = call(
        domain, "exchange_trip_money", amount=100, from_currency="USD", to_currency="EUR"
    )
    assert fx["converted"] == 90.91

    flight = call(
        domain,
        "hold_flight_seats",
        flight_id="F100",
        booked_by="A. Traveller",
        passengers=2,
        departure_date="2026-12-01",
        budget=400,
        currency="USD",
    )
    hotel = call(
        domain,
        "hold_hotel_rooms",
        hotel_id="H10",
        booked_by="A. Traveller",
        guests=3,
        check_in="2026-12-01",
        check_out="2026-12-03",
        budget=600,
        currency="EUR",
    )
    assert flight["booking_id"] == "B-1"
    assert hotel["booking_id"] == "B-2"
    assert hotel["cost"] == 581.82
    assert len(call(domain, "review_trip_bookings", booked_by="A. Traveller")) == 2
    call(domain, "release_trip_booking", booking_id="B-2")
    assert call(domain, "show_trip_booking", booking_id="B-2")["status"] == "cancelled"
    assert domain.state["flights"]["F200"] == decoy_flight
    assert domain.state["hotels"]["H20"] == decoy_hotel

    assert_goals(
        domain,
        [
            {"check": "count", "path": "bookings", "expected": 2},
            {"check": "exists", "path": "bookings.B-1"},
            {"check": "absent", "path": "bookings.B-3"},
            {"check": "value", "path": "bookings.B-2.status", "expected": "cancelled"},
            {"check": "value", "path": "flights.F100.seats_available", "expected": 6},
            {"check": "value", "path": "hotels.H10.rooms_available", "expected": 3},
            {"check": "value", "path": "flights.F200.seats_available", "expected": 5},
        ],
    )


def test_travel_rejects_over_budget_or_wrong_date_without_inventory_loss() -> None:
    domain = TravelDomain()
    message = assert_error(
        domain,
        "hold_flight_seats",
        flight_id="F100",
        booked_by="Budget Tester",
        passengers=2,
        departure_date="2026-12-01",
        budget=300,
        currency="USD",
    )
    assert "exceeds budget" in message
    assert_error(
        domain,
        "hold_flight_seats",
        flight_id="F100",
        booked_by="Date Tester",
        passengers=1,
        departure_date="2026-12-02",
    )
    assert_error(domain, "release_trip_booking", booking_id="missing")


def test_microblog_complete_auth_draft_publish_like_follow_flow() -> None:
    domain = MicroblogDomain()
    decoy = copy.deepcopy(domain.state["posts"]["M-2"])
    assert call(domain, "current_author")["authenticated"] is False
    call(domain, "open_author_session", handle="alice", auth_key="orchid")
    draft = call(domain, "stash_post_draft", text="First draft")
    assert draft["draft_id"] == "D-1"
    assert len(call(domain, "list_author_drafts")) == 1
    call(domain, "rewrite_post_draft", draft_id="D-1", text="A quieter harbour morning")
    published = call(domain, "publish_stashed_post", draft_id="D-1")
    assert published["post_id"] == "M-3"
    call(domain, "appreciate_micropost", post_id="M-1")
    call(domain, "subscribe_to_author", handle="bob")
    alice_feed = call(domain, "read_microblog_feed", author="alice")
    assert [post["post_id"] for post in alice_feed] == ["M-3"]
    assert domain.state["posts"]["M-2"] == decoy

    assert_goals(
        domain,
        [
            {"check": "count", "path": "drafts", "expected": 0},
            {"check": "absent", "path": "drafts.D-1"},
            {"check": "exists", "path": "posts.M-3"},
            {"check": "value", "path": "posts.M-3.text", "expected": "A quieter harbour morning"},
            {"check": "count", "path": "posts.M-1.liked_by", "expected": 1},
            {"check": "value", "path": "accounts.alice.following.0", "expected": "bob"},
            {"check": "count", "path": "posts.M-2.liked_by", "expected": 0},
        ],
    )
    call(domain, "close_author_session")
    assert call(domain, "current_author")["handle"] is None


def test_microblog_auth_and_duplicate_like_errors_are_atomic() -> None:
    domain = MicroblogDomain()
    assert_error(domain, "stash_post_draft", text="not authenticated")
    assert_error(domain, "open_author_session", handle="alice", auth_key="wrong")
    call(domain, "open_author_session", handle="alice", auth_key="orchid")
    call(domain, "appreciate_micropost", post_id="M-1")
    assert_error(domain, "appreciate_micropost", post_id="M-1")
    assert_error(domain, "discard_post_draft", draft_id="D-404")


def test_messaging_complete_login_contact_send_receive_read_flow() -> None:
    domain = MessagingDomain()
    untouched = copy.deepcopy(domain.state["messages"])
    call(domain, "enter_chat_account", username="alice", passcode="maple")
    assert call(domain, "chat_session_owner")["username"] == "alice"
    assert call(domain, "find_chat_people", query="bo") == ["bob"]
    sent = call(domain, "send_chat_note", recipient="bob", body="Meet at the north entrance")
    assert sent["message_id"] == "M-3"
    assert len(call(domain, "fetch_chat_outbox")) == 1
    call(domain, "leave_chat_account")
    call(domain, "enter_chat_account", username="bob", passcode="river")
    inbox = call(domain, "fetch_chat_inbox", unread_only=True)
    assert {message["message_id"] for message in inbox} == {"M-2", "M-3"}
    assert call(domain, "mark_chat_note_read", message_id="M-3")["already_read"] is False
    assert call(domain, "show_chat_note", message_id="M-3")["read"] is True
    remaining = call(domain, "fetch_chat_inbox", unread_only=True)
    assert [message["message_id"] for message in remaining] == ["M-2"]
    assert domain.state["messages"]["M-1"] == untouched["M-1"]
    assert domain.state["messages"]["M-2"] == untouched["M-2"]

    assert_goals(
        domain,
        [
            {"check": "count", "path": "messages", "expected": 3},
            {"check": "exists", "path": "messages.M-3"},
            {"check": "value", "path": "messages.M-3.read", "expected": True},
            {"check": "value", "path": "messages.M-1.read", "expected": False},
            {"check": "value", "path": "messages.M-2.read", "expected": False},
        ],
    )


def test_messaging_rejects_unknown_recipient_and_cross_account_reads() -> None:
    domain = MessagingDomain()
    assert_error(domain, "send_chat_note", recipient="bob", body="login first")
    call(domain, "enter_chat_account", username="alice", passcode="maple")
    assert_error(domain, "send_chat_note", recipient="nobody", body="hello")
    assert_error(domain, "show_chat_note", message_id="M-2")
    assert_error(domain, "mark_chat_note_read", message_id="M-2")


def test_ticketing_complete_create_edit_assign_resolve_reopen_flow() -> None:
    domain = TicketingDomain()
    decoy = copy.deepcopy(domain.state["tickets"]["T-1000"])
    created = call(
        domain,
        "file_support_ticket",
        title="VPN drops hourly",
        details="The tunnel disconnects near the start of each hour.",
        requester="chen",
        priority="high",
    )
    assert created["ticket_id"] == "T-1001"
    assert call(domain, "show_support_ticket", ticket_id="T-1001")["status"] == "open"
    call(domain, "assign_support_ticket", ticket_id="T-1001", agent="sam")
    call(
        domain,
        "revise_support_ticket",
        ticket_id="T-1001",
        title="VPN drops every hour",
        priority="urgent",
    )
    call(
        domain,
        "append_ticket_note",
        ticket_id="T-1001",
        author="sam",
        note="Reproduced on the test gateway.",
    )
    found = call(domain, "search_support_tickets", status="in_progress", requester="chen")
    assert [ticket["ticket_id"] for ticket in found] == ["T-1001"]
    call(
        domain,
        "close_support_ticket",
        ticket_id="T-1001",
        agent="riley",
        resolution="Restarted the gateway.",
    )
    call(domain, "reopen_support_ticket", ticket_id="T-1001")
    assert_goals(
        domain,
        [{"check": "absent", "path": "tickets.T-1001.resolution"}],
    )
    call(
        domain,
        "close_support_ticket",
        ticket_id="T-1001",
        agent="riley",
        resolution="Replaced the failing network cable.",
    )
    assert domain.state["tickets"]["T-1000"] == decoy

    assert_goals(
        domain,
        [
            {"check": "count", "path": "tickets", "expected": 2},
            {"check": "value", "path": "tickets.T-1001.status", "expected": "resolved"},
            {"check": "value", "path": "tickets.T-1001.priority", "expected": "urgent"},
            {"check": "exists", "path": "tickets.T-1001.resolution"},
            {"check": "count", "path": "tickets.T-1001.notes", "expected": 1},
            {"check": "absent", "path": "tickets.T-1000.resolution"},
            {"check": "value", "path": "tickets.T-1000.status", "expected": "open"},
        ],
    )


def test_ticketing_rejects_resolved_edits_and_unknown_agents_atomically() -> None:
    domain = TicketingDomain()
    assert_error(domain, "assign_support_ticket", ticket_id="T-1000", agent="stranger")
    call(
        domain,
        "close_support_ticket",
        ticket_id="T-1000",
        agent="sam",
        resolution="Replaced the printer power supply.",
    )
    assert_error(domain, "revise_support_ticket", ticket_id="T-1000", priority="low")
    assert_error(
        domain,
        "close_support_ticket",
        ticket_id="T-1000",
        agent="sam",
        resolution="Duplicate close",
    )
