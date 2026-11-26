"""Outcome-verified microblog, messaging, and ticketing stateful tasks."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..envs.base import build_env
from ..envs.mock_fc import MessagingDomain, MicroblogDomain, TicketingDomain
from ..rewards import dispatch_reward
from .schema_distractors import with_tool_distractors
from .schema_styles import apply_schema_style, mark_canonical
from .stateful_tasks import GeneratedTask, _c, _inject_fault, _s, _tool, to_task, verify_task

SOURCE = "social_ops_mt"
LICENSE = "generated (repo-internal, no third-party data)"


def _enum(description: str, *values: str) -> dict[str, Any]:
    return {"type": "string", "description": description, "enum": list(values)}


MICROBLOG_TOOLS = [
    _tool("open_author_session", MicroblogDomain.TOOL_DESCRIPTIONS["open_author_session"],
          {"handle": _s("Author handle."), "auth_key": _s("Account authentication key.")},
          ["handle", "auth_key"]),
    _tool("close_author_session", MicroblogDomain.TOOL_DESCRIPTIONS["close_author_session"], {}, []),
    _tool("current_author", MicroblogDomain.TOOL_DESCRIPTIONS["current_author"], {}, []),
    _tool("stash_post_draft", MicroblogDomain.TOOL_DESCRIPTIONS["stash_post_draft"],
          {"text": _s("Draft text, at most 280 characters.")}, ["text"]),
    _tool("list_author_drafts", MicroblogDomain.TOOL_DESCRIPTIONS["list_author_drafts"], {}, []),
    _tool("rewrite_post_draft", MicroblogDomain.TOOL_DESCRIPTIONS["rewrite_post_draft"],
          {"draft_id": _s("Owned draft identifier."), "text": _s("Replacement draft text.")},
          ["draft_id", "text"]),
    _tool("discard_post_draft", MicroblogDomain.TOOL_DESCRIPTIONS["discard_post_draft"],
          {"draft_id": _s("Owned draft identifier.")}, ["draft_id"]),
    _tool("publish_stashed_post", MicroblogDomain.TOOL_DESCRIPTIONS["publish_stashed_post"],
          {"draft_id": _s("Owned draft identifier.")}, ["draft_id"]),
    _tool("appreciate_micropost", MicroblogDomain.TOOL_DESCRIPTIONS["appreciate_micropost"],
          {"post_id": _s("Public micropost identifier.")}, ["post_id"]),
    _tool("subscribe_to_author", MicroblogDomain.TOOL_DESCRIPTIONS["subscribe_to_author"],
          {"handle": _s("Known author to follow.")}, ["handle"]),
    _tool("read_microblog_feed", MicroblogDomain.TOOL_DESCRIPTIONS["read_microblog_feed"],
          {"author": _s("Optional author filter.")}, []),
]
MICROBLOG_WRITES = {
    "open_author_session", "close_author_session", "stash_post_draft", "rewrite_post_draft",
    "discard_post_draft", "publish_stashed_post", "appreciate_micropost", "subscribe_to_author",
}

MESSAGING_TOOLS = [
    _tool("enter_chat_account", MessagingDomain.TOOL_DESCRIPTIONS["enter_chat_account"],
          {"username": _s("Messaging account name."), "passcode": _s("Account passcode.")},
          ["username", "passcode"]),
    _tool("leave_chat_account", MessagingDomain.TOOL_DESCRIPTIONS["leave_chat_account"], {}, []),
    _tool("chat_session_owner", MessagingDomain.TOOL_DESCRIPTIONS["chat_session_owner"], {}, []),
    _tool("find_chat_people", MessagingDomain.TOOL_DESCRIPTIONS["find_chat_people"],
          {"query": _s("Optional contact-name substring.")}, []),
    _tool("send_chat_note", MessagingDomain.TOOL_DESCRIPTIONS["send_chat_note"],
          {"recipient": _s("Known contact name."), "body": _s("Private message text.")},
          ["recipient", "body"]),
    _tool("fetch_chat_inbox", MessagingDomain.TOOL_DESCRIPTIONS["fetch_chat_inbox"],
          {"unread_only": {"type": "boolean", "default": False}}, []),
    _tool("fetch_chat_outbox", MessagingDomain.TOOL_DESCRIPTIONS["fetch_chat_outbox"], {}, []),
    _tool("mark_chat_note_read", MessagingDomain.TOOL_DESCRIPTIONS["mark_chat_note_read"],
          {"message_id": _s("Received message identifier.")}, ["message_id"]),
    _tool("show_chat_note", MessagingDomain.TOOL_DESCRIPTIONS["show_chat_note"],
          {"message_id": _s("Accessible message identifier.")}, ["message_id"]),
]
MESSAGING_WRITES = {"enter_chat_account", "leave_chat_account", "send_chat_note", "mark_chat_note_read"}

TICKETING_TOOLS = [
    _tool("file_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["file_support_ticket"],
          {"title": _s("Short issue title."), "details": _s("Full issue description."),
           "requester": _s("Requester name."),
           "priority": _enum("Initial priority.", "low", "normal", "high", "urgent")},
          ["title", "details", "requester"]),
    _tool("show_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["show_support_ticket"],
          {"ticket_id": _s("Support ticket identifier.")}, ["ticket_id"]),
    _tool("search_support_tickets", TicketingDomain.TOOL_DESCRIPTIONS["search_support_tickets"],
          {"status": _enum("Optional workflow state.", "open", "in_progress", "resolved"),
           "requester": _s("Optional exact requester.")}, []),
    _tool("revise_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["revise_support_ticket"],
          {"ticket_id": _s("Unresolved ticket identifier."), "title": _s("Optional replacement title."),
           "details": _s("Optional replacement details."),
           "priority": _enum("Optional replacement priority.", "low", "normal", "high", "urgent")},
          ["ticket_id"]),
    _tool("assign_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["assign_support_ticket"],
          {"ticket_id": _s("Unresolved ticket identifier."), "agent": _s("Known support agent.")},
          ["ticket_id", "agent"]),
    _tool("append_ticket_note", TicketingDomain.TOOL_DESCRIPTIONS["append_ticket_note"],
          {"ticket_id": _s("Unresolved ticket identifier."), "author": _s("Note author."),
           "note": _s("Note text.")}, ["ticket_id", "author", "note"]),
    _tool("close_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["close_support_ticket"],
          {"ticket_id": _s("Unresolved ticket identifier."), "agent": _s("Resolving support agent."),
           "resolution": _s("Resolution summary.")}, ["ticket_id", "agent", "resolution"]),
    _tool("reopen_support_ticket", TicketingDomain.TOOL_DESCRIPTIONS["reopen_support_ticket"],
          {"ticket_id": _s("Resolved ticket identifier.")}, ["ticket_id"]),
]
TICKETING_WRITES = {
    "file_support_ticket", "revise_support_ticket", "assign_support_ticket", "append_ticket_note",
    "close_support_ticket", "reopen_support_ticket",
}


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


def _microblog_state(rng: random.Random, handle: str, other: str) -> dict[str, Any]:
    target_code = rng.randint(1000, 9999)
    decoy_code = target_code + 1
    return {
        "accounts": {
            handle: {"auth_key": rng.choice(["orchid", "cobalt", "willow", "ember"]), "following": []},
            other: {"auth_key": rng.choice(["harbour", "meadow", "cinder"]), "following": []},
        },
        "active_user": None,
        "drafts": {
            "D-1": {"author": handle, "text": f"Field notes {target_code}: first version"},
            "D-2": {"author": handle, "text": f"Field notes {decoy_code}: keep this version"},
        },
        "posts": {
            "M-1": {"author": other, "text": f"Harbour study {rng.randint(100, 999)}", "liked_by": []},
            "M-2": {"author": other, "text": f"Train journal {rng.randint(100, 999)}", "liked_by": []},
        },
        "next_draft_id": 3,
        "next_post_id": 3,
    }


def microblog_rewrite_publish(rng: random.Random, level: int) -> DomainTask:
    handle, other = "alice", "bob"
    state = _microblog_state(rng, handle, other)
    auth_key = state["accounts"][handle]["auth_key"]
    target_code = state["drafts"]["D-1"]["text"].split()[2]
    post_code = state["posts"]["M-1"]["text"].rsplit(" ", 1)[1]
    new_text = f"Published field report {rng.randint(10000, 99999)}: observations verified"
    plan = [[_c("open_author_session", handle=handle, auth_key=auth_key)]]
    plan += _reads(rng, [
        _c("current_author"), _c("list_author_drafts"), _c("read_microblog_feed", author=other),
    ])
    plan += [
        [_c("rewrite_post_draft", draft_id="D-1", text=new_text)],
        [_c("publish_stashed_post", draft_id="D-1")],
        [_c("subscribe_to_author", handle=other)],
        [_c("read_microblog_feed", author=handle)],
    ]
    goal = [
        {"check": "absent", "path": "drafts.D-1", "name": "target_draft_published"},
        _check("drafts.D-2", copy.deepcopy(state["drafts"]["D-2"]), "unchanged_decoy_draft"),
        _check("posts.M-3.author", handle, "new_post_author"),
        _check("posts.M-3.text", new_text, "new_post_text"),
        _check("posts.M-3.liked_by", [], "new_post_likes"),
        _check("posts.M-1", copy.deepcopy(state["posts"]["M-1"]), "unchanged_post_one"),
        _check("posts.M-2", copy.deepcopy(state["posts"]["M-2"]), "unchanged_post_two"),
        _check(f"accounts.{handle}.following", [other], "subscription"),
        _check("posts", 3, "post_count", "length"),
        _check("next_post_id", 4, "next_post"),
    ]
    user = (
        f"Sign in as {handle} with key {auth_key}. Inspect your session, drafts, and {other}'s feed. Rewrite the "
        f"draft containing {target_code} to exactly '{new_text}', publish that draft, and subscribe to {other}. "
        f"The feed post numbered {post_code} and every other draft/post must remain unchanged. Read your own feed "
        "afterwards and report the new post id."
    )
    return DomainTask(
        "microblog", "rewrite_publish", user, MICROBLOG_TOOLS, state, goal, plan,
        final_answer="M-3", wrong_tool="publish_stashed_post", wrong_arguments={"draft_id": "D-2"},
        decoy_call=_c("rewrite_post_draft", draft_id="D-2", text="Unrequested decoy edit"),
    )


def microblog_draft_reactions(rng: random.Random, level: int) -> DomainTask:
    handle, other = "alice", "bob"
    state = _microblog_state(rng, handle, other)
    auth_key = state["accounts"][handle]["auth_key"]
    discard_code = state["drafts"]["D-2"]["text"].split()[2]
    post_code = state["posts"]["M-1"]["text"].rsplit(" ", 1)[1]
    new_text = f"Private launch idea {rng.randint(10000, 99999)}"
    plan = [[_c("open_author_session", handle=handle, auth_key=auth_key)]]
    plan += _reads(rng, [_c("list_author_drafts"), _c("read_microblog_feed", author=other)])
    plan += [
        [_c("stash_post_draft", text=new_text)],
        [_c("appreciate_micropost", post_id="M-1")],
        [_c("discard_post_draft", draft_id="D-2")],
        [_c("subscribe_to_author", handle=other)],
        [_c("list_author_drafts")],
    ]
    goal = [
        _check("drafts.D-3", {"author": handle, "text": new_text}, "stashed_draft"),
        {"check": "absent", "path": "drafts.D-2", "name": "selected_draft_discarded"},
        _check("drafts.D-1", copy.deepcopy(state["drafts"]["D-1"]), "unchanged_decoy_draft"),
        _check("drafts", 2, "remaining_drafts", "length"),
        _check("posts.M-1.liked_by", [handle], "post_appreciated"),
        _check("posts.M-2", copy.deepcopy(state["posts"]["M-2"]), "unchanged_decoy_post"),
        _check(f"accounts.{handle}.following", [other], "subscription"),
        _check("next_draft_id", 4, "next_draft"),
        _check("next_post_id", 3, "no_new_public_post"),
    ]
    user = (
        f"Sign in as {handle} using {auth_key}; list your drafts and {other}'s feed. Save a new private draft with "
        f"exact text '{new_text}', appreciate the feed post numbered {post_code}, discard only the draft containing "
        f"{discard_code}, and follow {other}. Keep the other draft and post unchanged. Finally list your drafts and "
        "report how many remain."
    )
    return DomainTask(
        "microblog", "draft_reactions", user, MICROBLOG_TOOLS, state, goal, plan,
        final_answer="2", wrong_tool="discard_post_draft", wrong_arguments={"draft_id": "D-1"},
        decoy_call=_c("appreciate_micropost", post_id="M-2"),
    )


def _messaging_state(rng: random.Random, active: str) -> dict[str, Any]:
    passcodes = {name: rng.choice(["maple", "river", "ember", "cobalt"]) for name in ("alice", "bob", "carol")}
    contacts = {
        "alice": ["bob", "carol"], "bob": ["alice", "carol"], "carol": ["alice", "bob"],
    }
    if active == "alice":
        messages = {
            "M-1": {"sender": "bob", "recipient": "alice", "body": f"Review packet {rng.randint(1000, 9999)}",
                    "read": False},
            "M-2": {"sender": "carol", "recipient": "alice", "body": f"Keep unread {rng.randint(1000, 9999)}",
                    "read": False},
        }
    else:
        messages = {
            "M-1": {"sender": "carol", "recipient": "bob", "body": f"Decoy note {rng.randint(1000, 9999)}",
                    "read": False},
            "M-2": {"sender": "alice", "recipient": "bob", "body": f"Action note {rng.randint(1000, 9999)}",
                    "read": False},
        }
    return {
        "accounts": {name: {"passcode": passcodes[name], "contacts": contacts[name]} for name in contacts},
        "active_user": None,
        "messages": messages,
        "next_message_id": 3,
    }


def messaging_read_reply(rng: random.Random, level: int) -> DomainTask:
    state = _messaging_state(rng, "alice")
    passcode = state["accounts"]["alice"]["passcode"]
    packet_code = state["messages"]["M-1"]["body"].rsplit(" ", 1)[1]
    reply = f"Packet {packet_code} approved at checkpoint {rng.randint(10, 99)}"
    plan = [[_c("enter_chat_account", username="alice", passcode=passcode)]]
    plan += _reads(rng, [
        _c("find_chat_people", query="bob"), _c("fetch_chat_inbox", unread_only=True), _c("fetch_chat_outbox"),
    ])
    plan += [
        [_c("show_chat_note", message_id="M-1")],
        [_c("mark_chat_note_read", message_id="M-1")],
        [_c("send_chat_note", recipient="bob", body=reply)],
        [_c("fetch_chat_outbox")],
    ]
    goal = [
        _check("messages.M-1.read", True, "target_read"),
        _check("messages.M-2", copy.deepcopy(state["messages"]["M-2"]), "decoy_unread"),
        _check("messages.M-3", {"sender": "alice", "recipient": "bob", "body": reply, "read": False}, "reply_sent"),
        _check("messages", 3, "message_count", "length"),
        _check("next_message_id", 4, "next_message"),
        _check("active_user", "alice", "session_owner"),
    ]
    user = (
        f"Log in as alice with passcode {passcode}. Find bob in your contacts and inspect your unread inbox and "
        f"outbox. Open Bob's packet message numbered {packet_code}, mark only that message read, and send Bob exactly "
        f"'{reply}'. Leave Carol's message unread. Review your outbox and report the new message id."
    )
    return DomainTask(
        "messaging", "read_reply", user, MESSAGING_TOOLS, state, goal, plan,
        final_answer="M-3", wrong_tool="send_chat_note", wrong_arguments={"recipient": "carol"},
        decoy_call=_c("mark_chat_note_read", message_id="M-2"),
    )


def messaging_two_notes(rng: random.Random, level: int) -> DomainTask:
    state = _messaging_state(rng, "bob")
    passcode = state["accounts"]["bob"]["passcode"]
    action_code = state["messages"]["M-2"]["body"].rsplit(" ", 1)[1]
    first_body = f"Alice update {rng.randint(1000, 9999)}"
    second_body = f"Carol update {rng.randint(1000, 9999)}"
    plan = [[_c("enter_chat_account", username="bob", passcode=passcode)]]
    plan += _reads(rng, [
        _c("find_chat_people", query=""), _c("fetch_chat_inbox", unread_only=True), _c("fetch_chat_outbox"),
    ])
    plan += [
        [_c("send_chat_note", recipient="alice", body=first_body)],
        [_c("send_chat_note", recipient="carol", body=second_body)],
        [_c("mark_chat_note_read", message_id="M-2")],
        [_c("fetch_chat_outbox")],
    ]
    goal = [
        _check("messages.M-1", copy.deepcopy(state["messages"]["M-1"]), "decoy_unread"),
        _check("messages.M-2.read", True, "target_read"),
        _check("messages.M-3", {"sender": "bob", "recipient": "alice", "body": first_body, "read": False},
               "alice_note"),
        _check("messages.M-4", {"sender": "bob", "recipient": "carol", "body": second_body, "read": False},
               "carol_note"),
        _check("messages", 4, "message_count", "length"),
        _check("next_message_id", 5, "next_message"),
    ]
    user = (
        f"Log in as bob using {passcode}; inspect contacts, unread inbox, and outbox. Send Alice exactly '{first_body}' "
        f"and Carol exactly '{second_body}', in that order. Mark only Alice's incoming action note numbered "
        f"{action_code} as read, leaving Carol's incoming note unread. Review the outbox and report how many messages "
        "you sent."
    )
    return DomainTask(
        "messaging", "two_notes", user, MESSAGING_TOOLS, state, goal, plan,
        final_answer="2", wrong_tool="send_chat_note", wrong_arguments={"recipient": "alice"},
        decoy_call=_c("mark_chat_note_read", message_id="M-1"),
    )


def _ticket(title: str, details: str, requester: str, priority: str, status: str = "open") -> dict[str, Any]:
    return {
        "title": title, "details": details, "requester": requester, "priority": priority,
        "status": status, "assignee": None, "resolution": None, "resolved_by": None, "notes": [],
    }


def ticketing_triage(rng: random.Random, level: int) -> DomainTask:
    requester = rng.choice(["alice", "morgan", "chen"])
    incident = rng.randint(1000, 9999)
    target = _ticket(f"Gateway incident {incident}", "Intermittent gateway timeouts.", requester, "normal")
    decoy = _ticket(f"Printer request {incident + 1}", "Replace the reception printer cable.", requester, "low")
    agents = rng.sample(["sam", "riley", "alex"], 2)
    agent, wrong_agent = agents
    details = f"Incident {incident}: affected two gateway nodes; failover is active."
    note = f"Triage note {rng.randint(100, 999)}: monitor failover for one hour."
    state = {"agents": agents, "tickets": {"T-1000": target, "T-1001": decoy}, "next_ticket_id": 1002}
    plan = [[_c("search_support_tickets", status="open", requester=requester)]]
    plan += _reads(rng, [_c("show_support_ticket", ticket_id="T-1000"),
                         _c("show_support_ticket", ticket_id="T-1001")])
    plan += [
        [_c("revise_support_ticket", ticket_id="T-1000", details=details, priority="urgent")],
        [_c("assign_support_ticket", ticket_id="T-1000", agent=agent)],
        [_c("append_ticket_note", ticket_id="T-1000", author=agent, note=note)],
        [_c("show_support_ticket", ticket_id="T-1000")],
    ]
    goal = [
        _check("tickets.T-1000.details", details, "details_revised"),
        _check("tickets.T-1000.priority", "urgent", "priority_revised"),
        _check("tickets.T-1000.assignee", agent, "assigned"),
        _check("tickets.T-1000.status", "in_progress", "in_progress"),
        _check("tickets.T-1000.notes", [{"author": agent, "text": note}], "note_added"),
        _check("tickets.T-1001", copy.deepcopy(decoy), "unchanged_decoy_ticket"),
        _check("tickets", 2, "ticket_count", "length"),
        _check("next_ticket_id", 1002, "no_new_ticket"),
    ]
    user = (
        f"Search open tickets requested by {requester}, then inspect both. For the gateway incident numbered "
        f"{incident}, replace its details with '{details}', set priority urgent, assign {agent}, and append exactly "
        f"'{note}' as {agent}. Leave the printer ticket completely unchanged. Show the updated ticket and report its id."
    )
    return DomainTask(
        "ticketing", "triage_existing", user, TICKETING_TOOLS, state, goal, plan,
        final_answer="T-1000", wrong_tool="assign_support_ticket", wrong_arguments={"agent": wrong_agent},
        decoy_call=_c("revise_support_ticket", ticket_id="T-1001", priority="high"),
    )


def ticketing_create_resolve(rng: random.Random, level: int) -> DomainTask:
    requester = rng.choice(["dana", "omar", "priya"])
    incident = rng.randint(10000, 99999)
    decoy_open = _ticket(f"Open decoy {incident}", "Leave this request untouched.", requester, "normal")
    decoy_resolved = _ticket(f"Resolved decoy {incident + 1}", "Already handled.", "someone", "low", "resolved")
    decoy_resolved.update({"assignee": "sam", "resolution": "Completed earlier", "resolved_by": "sam"})
    agents = ["sam", "riley"]
    agent = rng.choice(agents)
    title = f"Access outage {incident}"
    initial_details = f"User report {incident}: access is unavailable."
    details = f"Confirmed outage {incident}: credentials were expired and have been renewed."
    note = f"Resolution evidence {rng.randint(100, 999)} attached."
    resolution = f"Renewed credentials and verified access for incident {incident}."
    state = {
        "agents": agents,
        "tickets": {"T-1000": decoy_open, "T-1001": decoy_resolved},
        "next_ticket_id": 1002,
    }
    plan = [
        [_c("search_support_tickets", requester=requester)],
        [_c("file_support_ticket", title=title, details=initial_details, requester=requester, priority="normal")],
        [_c("show_support_ticket", ticket_id="T-1002")],
        [_c("revise_support_ticket", ticket_id="T-1002", details=details, priority="high")],
        [_c("assign_support_ticket", ticket_id="T-1002", agent=agent)],
        [_c("append_ticket_note", ticket_id="T-1002", author=agent, note=note)],
        [_c("close_support_ticket", ticket_id="T-1002", agent=agent, resolution=resolution)],
        [_c("show_support_ticket", ticket_id="T-1002")],
    ]
    goal = [
        _check("tickets.T-1002.title", title, "created_title"),
        _check("tickets.T-1002.details", details, "revised_details"),
        _check("tickets.T-1002.requester", requester, "requester"),
        _check("tickets.T-1002.priority", "high", "priority"),
        _check("tickets.T-1002.status", "resolved", "resolved"),
        _check("tickets.T-1002.assignee", agent, "assignee"),
        _check("tickets.T-1002.resolution", resolution, "resolution"),
        _check("tickets.T-1002.resolved_by", agent, "resolved_by"),
        _check("tickets.T-1002.notes", [{"author": agent, "text": note}], "note"),
        _check("tickets.T-1000", copy.deepcopy(decoy_open), "unchanged_open_decoy"),
        _check("tickets.T-1001", copy.deepcopy(decoy_resolved), "unchanged_resolved_decoy"),
        _check("tickets", 3, "ticket_count", "length"),
        _check("next_ticket_id", 1003, "next_ticket"),
    ]
    user = (
        f"Search {requester}'s existing requests, then file a normal-priority ticket titled '{title}' with details "
        f"'{initial_details}'. Inspect the new ticket, revise its details to '{details}' and priority to high, assign "
        f"{agent}, add {agent}'s note '{note}', and resolve it as {agent} with resolution '{resolution}'. Do not alter "
        "either existing ticket. Show the final ticket and report its id."
    )
    return DomainTask(
        "ticketing", "create_resolve", user, TICKETING_TOOLS, state, goal, plan,
        final_answer="T-1002", wrong_tool="close_support_ticket",
        wrong_arguments={"resolution": "Incorrect unrequested resolution"},
        decoy_call=_c("append_ticket_note", ticket_id="T-1000", author=agent, note="Unrequested decoy note"),
    )


DOMAIN_GENERATORS: dict[str, list[tuple[Callable[[random.Random, int], DomainTask], set[str]]]] = {
    "microblog": [(microblog_rewrite_publish, MICROBLOG_WRITES), (microblog_draft_reactions, MICROBLOG_WRITES)],
    "messaging": [(messaging_read_reply, MESSAGING_WRITES), (messaging_two_notes, MESSAGING_WRITES)],
    "ticketing": [(ticketing_triage, TICKETING_WRITES), (ticketing_create_resolve, TICKETING_WRITES)],
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
    return [wrong, copy.deepcopy(gen.plan) + [[copy.deepcopy(gen.decoy_call)]]]


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
