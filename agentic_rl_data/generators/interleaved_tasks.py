"""Outcome-verified dialogues with scripted, interleaved user instructions.

Only the first user message is visible in ``task.messages``.  Later requests
live in ``env.config.dialogue.stages`` and are released by the environment only
after the preceding goal, evidence, reply, and invariant gates pass.  Every
stage plan is computed against the state left by the previous stage.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from ..envs.base import build_env
from ..envs.mock_fc import DEFAULT_FAULT_MESSAGE
from ..rewards import dispatch_reward
from ..schema import validate_task
from .schema_distractors import with_tool_distractors
from .schema_styles import apply_schema_style, mark_canonical
from .stateful_tasks import (
    _CITIES,
    _STREETS,
    _SYMBOLS,
    FS_TOOLS,
    FS_WRITES,
    RETAIL_TOOLS,
    RETAIL_WRITES,
    TRADE_TOOLS,
    TRADE_WRITES,
    _c,
    _prices,
    _reads,
    _retail_lookup,
    _retail_world,
    _trade_goal,
)

SOURCE = "interleaved_mt"
LICENSE = "generated (repo-internal, no third-party data)"
MAX_TURNS_CAP = 24
DOMAINS = ("filesystem", "trading", "retail")

SYSTEM_PROMPT = (
    "You are an autonomous agent completing a multi-request task with the provided tools. "
    "Read relevant state before changing it, check tool errors and retry transient failures. "
    "After each request is genuinely complete, reply briefly so the user can continue. "
    "Later user requests may append to, correct, or replace earlier goals: preserve prior work "
    "when asked, undo superseded effects when necessary, and never imagine future instructions. "
    "If a requested value is required, put exactly that value in <answer></answer> in the final reply."
)


@dataclass
class _Stage:
    kind: str
    user: str
    goal: list[dict[str, Any]]
    plan: list[list[dict[str, Any]]]
    invariants: list[dict[str, Any]] = field(default_factory=list)
    must_change: list[dict[str, Any]] = field(default_factory=list)
    reply: str = "none"
    slot_keywords: list[Any] = field(default_factory=list)
    content: str = "Done."


def _check_names(checks: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [copy.deepcopy(check) for check in checks if str(check.get("name")) in names]


def _check_paths(checks: list[dict[str, Any]], paths: set[str]) -> list[dict[str, Any]]:
    return [copy.deepcopy(check) for check in checks if str(check.get("path")) in paths]


def _file(path: str, content: str, name: str | None = None) -> dict[str, Any]:
    return {"check": "file_content", "path": path, "expected": content, "name": name or f"file:{path}"}


def _no_file(path: str) -> dict[str, Any]:
    return {"check": "file_not_exists", "path": path, "name": f"absent:{path}"}


# ---------------------------------------------------------------------------
# filesystem: draft -> additions -> correction -> replacement goal
# ---------------------------------------------------------------------------


def _filesystem_dialogue(rng: random.Random, turn_count: int) -> tuple[dict[str, Any], list[_Stage], str]:
    topic = rng.choice(["aurora", "beacon", "cedar", "delta", "ember", "falcon"])
    brief = f"{topic} brief, revision {rng.randint(10, 99)}.\n"
    corrected = f"{topic} corrected draft, revision {rng.randint(100, 999)}.\n"
    final = f"{topic} final deliverable, revision {rng.randint(1000, 9999)}.\n"
    decoy = "Unrelated notes; preserve byte for byte.\n"
    paths = {"draft": "/workspace/draft.txt", "backup": "/workspace/draft.bak", "final": "/workspace/final.txt"}
    base = [_file("/workspace/brief.txt", brief, "brief"), _file("/workspace/decoy.txt", decoy, "decoy")]
    state = {
        "files": {"/workspace/brief.txt": brief, "/workspace/decoy.txt": decoy},
        "dirs": ["/", "/workspace"],
    }

    first_kind = "append" if turn_count == 3 else "initial"
    first_goal = base + [
        _file(paths["draft"], brief, "draft"),
        _file(paths["backup"], brief, "backup"),
    ]
    stages = [
        _Stage(
            first_kind,
            "Read /workspace/brief.txt, create /workspace/draft.txt with exactly that content, "
            "and also copy the draft to /workspace/draft.bak. Leave brief.txt and decoy.txt unchanged.",
            first_goal,
            [
                [_c("read_file", path="/workspace/brief.txt")],
                [_c("write_file", path=paths["draft"], content=brief)],
                [_c("copy_file", source=paths["draft"], destination=paths["backup"])],
            ],
            invariants=copy.deepcopy(base),
            must_change=_check_names(first_goal, {"draft", "backup"}),
        )
    ]
    current_goal = first_goal
    artifacts = [paths["draft"], paths["backup"]]

    for index in range(turn_count - 3):
        note = f"/workspace/note_{index + 1}.txt"
        note_content = f"{topic} appended note {index + 1}: {rng.randint(100, 999)}.\n"
        note_check = _file(note, note_content, f"note:{index + 1}")
        previous = copy.deepcopy(current_goal)
        current_goal = previous + [note_check]
        artifacts.append(note)
        stages.append(
            _Stage(
                "append",
                f"Also create {note} with exactly this content: {note_content!r}. "
                "Keep the draft, backup, and every earlier note unchanged.",
                current_goal,
                [[_c("read_file", path=paths["draft"])], [_c("write_file", path=note, content=note_content)]],
                invariants=previous,
                must_change=[note_check],
            )
        )

    preserved_notes = [check for check in current_goal if str(check.get("name", "")).startswith("note:")]
    correction_goal = base + [
        _file(paths["draft"], corrected, "draft"),
        _file(paths["backup"], corrected, "backup"),
    ] + preserved_notes
    stages.append(
        _Stage(
            "correction",
            f"Correction: both draft.txt and draft.bak should instead contain exactly {corrected!r}. "
            "Update both files and keep all appended notes and the two original files unchanged.",
            correction_goal,
            [
                [_c("read_file", path=paths["draft"])],
                [_c("write_file", path=paths["draft"], content=corrected)],
                [_c("write_file", path=paths["backup"], content=corrected)],
            ],
            invariants=base + preserved_notes,
            must_change=_check_names(correction_goal, {"draft", "backup"}),
        )
    )

    final_goal = base + [_file(paths["final"], final, "final")] + [_no_file(path) for path in artifacts]
    # These writes touch disjoint paths and can be issued in one assistant turn.
    final_plan = [[
        _c("write_file", path=paths["final"], content=final),
        *[_c("delete_file", path=path) for path in artifacts],
    ]]
    stages.append(
        _Stage(
            "change_goal",
            f"I have changed the goal. Keep only /workspace/final.txt containing exactly {final!r}; "
            "delete draft.txt, draft.bak, and every appended note. Preserve brief.txt and decoy.txt.",
            final_goal,
            final_plan,
            invariants=copy.deepcopy(base),
            must_change=[_file(paths["final"], final, "final")] + [_no_file(path) for path in artifacts],
        )
    )
    return state, stages, "filesystem_revision"


# ---------------------------------------------------------------------------
# trading: rebalance -> additions -> corrected quantity -> liquidation
# ---------------------------------------------------------------------------


def _trading_dialogue(rng: random.Random, turn_count: int) -> tuple[dict[str, Any], list[_Stage], str, str]:
    a, b, c = rng.sample(_SYMBOLS, 3)
    prices = _prices(rng, [a, b, c])
    cash = round(rng.uniform(15000, 30000), 2)
    holdings = {a: 8, b: 2, c: rng.randint(2, 7)}
    # The plan simulation mutates ``holdings``; keep the episode start snapshot.
    state = {"balance": cash, "holdings": dict(holdings), "prices": prices, "history": []}
    invariant = [
        {"check": "value", "path": f"holdings.{c}", "expected": holdings[c], "name": f"untouched:{c}"}
    ]

    stages: list[_Stage] = []
    first_goal = _trade_goal(holdings, cash)
    plan = [[_c("get_holdings")]] + _reads(rng, [_c("get_price", symbol=a)])
    plan.append([_c("sell", symbol=a, quantity=5)])
    holdings[a] -= 5
    cash = round(cash + 5 * prices[a], 2)
    first_goal = _trade_goal(holdings, cash)
    stages.append(
        _Stage(
            "append" if turn_count == 3 else "initial",
            f"Sell 5 shares of {a}. Keep my {b} and {c} positions unchanged for now; "
            "I may add another instruction after this is done.",
            first_goal,
            plan,
            invariants=copy.deepcopy(invariant),
            must_change=_check_names(first_goal, {"balance", f"holding:{a}"}),
        )
    )

    for index in range(turn_count - 3):
        if index % 2 == 0:
            amount = float(75 + 25 * index + rng.randint(0, 20))
            plan = [[_c("get_balance")], [_c("deposit", amount=amount)]]
            cash = round(cash + amount, 2)
            user = f"Also deposit exactly ${amount:.2f}; leave all stock positions as they are now."
            changed = {"balance"}
        else:
            plan = [[_c("get_price", symbol=b)], [_c("buy", symbol=b, quantity=1)]]
            holdings[b] += 1
            cash = round(cash - prices[b], 2)
            user = f"Also buy one more share of {b}, without altering the {a} or {c} positions."
            changed = {"balance", f"holding:{b}"}
        goal = _trade_goal(holdings, cash)
        stages.append(_Stage("append", user, goal, plan, invariants=copy.deepcopy(invariant),
                             must_change=_check_names(goal, changed)))

    correction_plan = [[_c("get_holdings")]] + _reads(rng, [_c("get_price", symbol=a)])
    correction_plan.append([_c("buy", symbol=a, quantity=2)])
    holdings[a] += 2
    cash = round(cash - 2 * prices[a], 2)
    correction_goal = _trade_goal(holdings, cash)
    stages.append(
        _Stage(
            "correction",
            f"Correction: I meant to sell only 3 {a} shares in total, not 5. Buy back 2 {a} shares "
            "to correct that earlier sale, while preserving every subsequent deposit or purchase.",
            correction_goal,
            correction_plan,
            invariants=copy.deepcopy(invariant),
            must_change=_check_names(correction_goal, {"balance", f"holding:{a}"}),
        )
    )

    final_plan: list[list[dict[str, Any]]] = [[_c("get_holdings")]]
    final_plan += _reads(rng, [_c("get_price", symbol=a), _c("get_price", symbol=b)])
    for symbol in (a, b):
        quantity = holdings[symbol]
        if quantity:
            final_plan.append([_c("sell", symbol=symbol, quantity=quantity)])
            cash = round(cash + quantity * prices[symbol], 2)
            holdings[symbol] = 0
    final_plan.append([_c("get_balance")])
    final_goal = _trade_goal(holdings, cash)
    answer = f"{cash:.2f}"
    stages.append(
        _Stage(
            "change_goal",
            f"New goal: liquidate all remaining {a} and {b} instead of keeping the corrected position. "
            f"Do not touch {c}. Tell me the final cash balance rounded to 2 decimals.",
            final_goal,
            final_plan,
            invariants=copy.deepcopy(invariant),
            must_change=_check_names(final_goal, {"balance", f"holding:{a}", f"holding:{b}"}),
            reply="answer_tag",
            content=f"Done. <answer>{answer}</answer>",
        )
    )
    return state, stages, "trading_rebalance", answer


# ---------------------------------------------------------------------------
# retail: address request -> additions -> correction -> restore and cancel
# ---------------------------------------------------------------------------


def _order_goal(orders: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for order_id, order in orders.items():
        checks.append({"check": "value", "path": f"orders.{order_id}.status",
                       "expected": order["status"], "name": f"status:{order_id}"})
        checks.append({"check": "value", "path": f"orders.{order_id}.address",
                       "expected": order["address"], "name": f"address:{order_id}"})
        for field_name in ("cancel_reason", "return_items", "exchange"):
            if field_name in order:
                checks.append({"check": "value", "path": f"orders.{order_id}.{field_name}",
                               "expected": copy.deepcopy(order[field_name]), "name": f"{field_name}:{order_id}"})
    return checks


def _item(state: Mapping[str, Any], order_id: str) -> dict[str, Any]:
    return state["orders"][order_id]["items"][0]


def _label(state: Mapping[str, Any], order_id: str) -> str:
    item = _item(state, order_id)
    return f"{item['options']['color']} {item['name']}"


def _new_address(rng: random.Random, old: str) -> str:
    value = old
    while value == old:
        value = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    return value


def _retail_dialogue(rng: random.Random, turn_count: int) -> tuple[dict[str, Any], list[_Stage], str]:
    state, meta = _retail_world(rng, 5, ["pending", "pending", "pending", "delivered", "delivered"])
    target, cancel_other, move_other, return_other, decoy = list(state["orders"])
    before = copy.deepcopy(state["orders"])
    expected = copy.deepcopy(before)
    address_one = _new_address(rng, meta["address"])
    address_two = _new_address(rng, address_one)
    address_three = _new_address(rng, meta["address"])
    stages: list[_Stage] = []

    changed = {f"orders.{target}.address"}
    expected[target]["address"] = address_one
    plan = _retail_lookup(meta, state, rng)
    plan.append([_c("modify_order_address", order_id=target, address=address_one)])
    goal = _order_goal(expected)
    stages.append(
        _Stage(
            "append" if turn_count == 3 else "initial",
            f"I'm {meta['name']} ({meta['email']}). Ship my pending order with the {_label(state, target)} "
            f"to {address_one}. Keep every other order unchanged.",
            goal,
            plan,
            invariants=[
                check for check in _order_goal(before) if str(check.get("path")) not in changed
            ],
            must_change=_check_paths(goal, changed),
        )
    )
    before = copy.deepcopy(expected)

    for index in range(turn_count - 3):
        if index == 0:
            order_id = cancel_other
            reason = "no longer needed"
            expected[order_id]["status"] = "cancelled"
            expected[order_id]["cancel_reason"] = reason
            plan = [[_c("cancel_pending_order", order_id=order_id, reason=reason)]]
            user = f"Also cancel my pending order with the {_label(state, order_id)} because it is {reason}."
            changed = {f"orders.{order_id}.status", f"orders.{order_id}.cancel_reason"}
        elif index == 1:
            order_id = return_other
            item_id = _item(state, order_id)["item_id"]
            expected[order_id]["status"] = "return requested"
            expected[order_id]["return_items"] = sorted([item_id])
            plan = [[_c("return_delivered_items", order_id=order_id, item_ids=[item_id])]]
            user = f"Also return the {_label(state, order_id)} from my delivered order."
            changed = {f"orders.{order_id}.status", f"orders.{order_id}.return_items"}
        else:
            order_id = move_other
            expected[order_id]["address"] = address_three
            plan = [[_c("modify_order_address", order_id=order_id, address=address_three)]]
            user = f"Additionally, send the pending {_label(state, order_id)} order to {address_three}."
            changed = {f"orders.{order_id}.address"}
        goal = _order_goal(expected)
        stages.append(
            _Stage(
                "append",
                user + " Preserve all changes already made and do not alter unrelated orders.",
                goal,
                plan,
                invariants=[check for check in _order_goal(before) if str(check.get("path")) not in changed],
                must_change=_check_paths(goal, changed),
            )
        )
        before = copy.deepcopy(expected)

    changed = {f"orders.{target}.address"}
    expected[target]["address"] = address_two
    goal = _order_goal(expected)
    stages.append(
        _Stage(
            "correction",
            f"Correction: the {_label(state, target)} order should go to {address_two}, not the first address. "
            "Update only that address and preserve all appended requests.",
            goal,
            [[_c("modify_order_address", order_id=target, address=address_two)]],
            invariants=[check for check in _order_goal(before) if str(check.get("path")) not in changed],
            must_change=_check_paths(goal, changed),
        )
    )
    before = copy.deepcopy(expected)

    reason = "ordered by mistake"
    changed = {f"orders.{target}.address", f"orders.{target}.status", f"orders.{target}.cancel_reason"}
    expected[target]["address"] = meta["address"]
    expected[target]["status"] = "cancelled"
    expected[target]["cancel_reason"] = reason
    goal = _order_goal(expected)
    stages.append(
        _Stage(
            "change_goal",
            f"I changed my mind about the {_label(state, target)} order: restore its original address "
            f"{meta['address']} and cancel it because it was {reason}. Keep all other completed requests intact, "
            f"and never touch the {_label(state, decoy)} order.",
            goal,
            [
                [_c("modify_order_address", order_id=target, address=meta["address"])],
                [_c("cancel_pending_order", order_id=target, reason=reason)],
            ],
            invariants=[check for check in _order_goal(before) if str(check.get("path")) not in changed],
            must_change=_check_paths(goal, changed),
        )
    )
    return state, stages, "retail_change_mind"


# ---------------------------------------------------------------------------
# assembly, faults, and real-env verification
# ---------------------------------------------------------------------------


def _inject_fault(stages: list[_Stage], rng: random.Random, writes: set[str]) -> tuple[dict[str, Any], tuple[int, int]] | None:
    seen: set[str] = set()
    candidates: list[tuple[int, int]] = []
    for stage_index, stage in enumerate(stages):
        for group_index, group in enumerate(stage.plan):
            for call in group:
                name = str(call["name"])
                if len(group) == 1 and name in writes and name not in seen:
                    candidates.append((stage_index, group_index))
                seen.add(name)
    if not candidates:
        return None
    stage_index, group_index = rng.choice(candidates)
    group = stages[stage_index].plan[group_index]
    tool = str(group[0]["name"])
    stages[stage_index].plan.insert(group_index + 1, copy.deepcopy(group))
    fault = {"tool": tool, "count": 1, "message": DEFAULT_FAULT_MESSAGE}
    return fault, (stage_index, group_index + 1)


def _make_task(
    *,
    task_id: str,
    domain: str,
    scenario: str,
    initial_state: dict[str, Any],
    stages: list[_Stage],
    fault: dict[str, Any] | None,
    retry_at: tuple[int, int] | None,
    slack: int,
    final_answer: str | None = None,
) -> dict[str, Any]:
    groups: list[list[dict[str, Any]]] = []
    turns: list[dict[str, Any]] = []
    stage_payloads: list[dict[str, Any]] = []
    retry_groups: list[int] = []
    for stage_index, stage in enumerate(stages):
        required: list[str] = []
        for group_index, group in enumerate(stage.plan):
            if retry_at == (stage_index, group_index):
                retry_groups.append(len(groups))
            groups.append(copy.deepcopy(group))
            turns.append({"content": "", "next_user": None})
            for call in group:
                if call["name"] not in required:
                    required.append(call["name"])
        next_user = stages[stage_index + 1].user if stage_index + 1 < len(stages) else None
        groups.append([])
        turns.append({"content": stage.content, "next_user": next_user})
        stage_payloads.append(
            {
                "kind": stage.kind,
                "goal": copy.deepcopy(stage.goal),
                "invariants": copy.deepcopy(stage.invariants),
                "required_tools": required,
                "must_change": copy.deepcopy(stage.must_change),
                "reply": stage.reply,
                "slot_keywords": copy.deepcopy(stage.slot_keywords),
                "next_user": next_user,
            }
        )

    expected = len(groups)
    if expected > MAX_TURNS_CAP:
        raise ValueError(f"dialogue needs {expected} turns, above cap {MAX_TURNS_CAP}")
    max_turns = min(MAX_TURNS_CAP, expected + slack)
    config: dict[str, Any] = {
        "impl": "stateful",
        "domain": domain,
        "initial_state": copy.deepcopy(initial_state),
        "goal": copy.deepcopy(stages[-1].goal),
        "max_turns": max_turns,
        "process_term": "efficiency",
        "dialogue": {"stages": stage_payloads},
    }
    if final_answer is not None:
        config["require_final_answer"] = True
        config["final_answer_mode"] = "answer_tag"
    if fault is not None:
        config["faults"] = [fault]
    tools = {"filesystem": FS_TOOLS, "trading": TRADE_TOOLS, "retail": RETAIL_TOOLS}[domain]
    return {
        "id": task_id,
        "source": SOURCE,
        "task_type": "mt_fc",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": stages[0].user}],
        "tools": copy.deepcopy(tools),
        "env": {"type": "mock_fc", "config": config},
        "reference": {
            "final_answer": final_answer,
            "final_answers": None,
            "gt_calls": groups,
            "gt_turns": turns,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": max_turns,
            "expected_turns": expected,
            "difficulty": float(len(stages) - 3),
            "pass_rate": None,
            "license": LICENSE,
            "gt_returns_source": "config",
            "group_id": task_id,
            "domain": domain,
            "scenario": scenario,
            "template": scenario,
            "user_turn_count": len(stages),
            "parallel_group_count": sum(1 for group in groups if len(group) > 1),
            "canonical_tools": [tool["name"] for tool in tools],
            "schema_style": "canonical",
            "n_faults": int(fault is not None),
            "retry_groups": retry_groups,
            "requires_answer": final_answer is not None,
        },
    }


def _reference_actions(task: Mapping[str, Any], skip: int | None = None) -> list[tuple[list[dict[str, Any]], str, str | None]]:
    actions = []
    for index, (group, turn) in enumerate(zip(task["reference"]["gt_calls"], task["reference"]["gt_turns"])):
        if index == skip:
            continue
        actions.append((copy.deepcopy(group), str(turn.get("content") or ""), turn.get("next_user")))
    return actions


def _run_actions(
    task: dict[str, Any],
    actions: list[tuple[list[dict[str, Any]], str, str | None]],
    *,
    finish: bool,
    check_next_user: bool = False,
) -> dict[str, Any]:
    env = build_env(task)
    env.reset(task)
    try:
        for index, (group, content, expected_next) in enumerate(actions):
            _, done, info = env.step({"content": content, "tool_calls": group})
            if check_next_user:
                actual_next = info.get("next_user_message")
                if actual_next != expected_next:
                    raise ValueError(
                        f"next_user mismatch at action {index}: expected {expected_next!r}, got {actual_next!r}"
                    )
                if done and index != len(actions) - 1:
                    raise ValueError(f"dialogue terminated early at action {index}")
            if done:
                break
        if finish and not env.done:
            env.step({"content": "All done.", "tool_calls": []})
        return env.trajectory()
    finally:
        env.close()


def verify_task(task: dict[str, Any], retry_groups: list[int] | None = None) -> dict[str, Any]:
    errors = validate_task(task)
    if errors:
        raise ValueError(f"schema: {errors[:3]}")
    good = _run_actions(task, _reference_actions(task), finish=False, check_next_user=True)
    if not good["done"]:
        raise ValueError("GT dialogue did not terminate")
    if (good["env_state"] or {}).get("goal_satisfied") is not True:
        raise ValueError("GT dialogue does not satisfy the final goal")
    score = float(dispatch_reward(task, good)["score"])
    if score < 1.0 - 1e-9:
        raise ValueError(f"GT dialogue scores {score:.3f} < 1.0")

    idle = _run_actions(task, [], finish=True)
    if float(dispatch_reward(task, idle)["score"]) != 0.0:
        raise ValueError("idle reply does not score zero")

    retries = retry_groups if retry_groups is not None else task["meta"].get("retry_groups", [])
    for index in retries:
        dropped = _run_actions(task, _reference_actions(task, skip=index), finish=True)
        if (dropped["env_state"] or {}).get("goal_satisfied") is True:
            raise ValueError("dropping the fault retry still satisfies the dialogue")
        if float(dispatch_reward(task, dropped)["score"]) >= 1.0 - 1e-9:
            raise ValueError("dropping the fault retry still earns full reward")

    answer = task["reference"].get("final_answer")
    if answer is not None:
        wrong_actions = _reference_actions(task)
        group, _, next_user = wrong_actions[-1]
        wrong_actions[-1] = (group, "All done.", next_user)
        wrong = _run_actions(task, wrong_actions, finish=True)
        if (wrong["env_state"] or {}).get("goal_satisfied") is True:
            raise ValueError("missing final answer still satisfies the dialogue")
    return {"score": score, "num_turns": good["num_turns"]}


def _quota_flags(n: int, probability: float, seed: int, label: str, eligible: list[int] | None = None) -> set[int]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label} probability must be in [0, 1]")
    candidates = list(range(n)) if eligible is None else list(eligible)
    count = min(len(candidates), int(n * probability + 0.5) if probability else 0)
    rng = random.Random(f"{seed}:{label}:quota")
    rng.shuffle(candidates)
    return set(candidates[:count])


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    domains: list[str] | None = None,
    turn_counts: list[int] | None = None,
    fault_prob: float = 0.25,
    schema_prob: float = 0.75,
    slack: int = 3,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield ``n`` deterministic tasks, stratified by domain and user turns."""
    if n < 0:
        raise ValueError("n must be non-negative")
    domains = list(domains or DOMAINS)
    unknown = sorted(set(domains) - set(DOMAINS))
    if unknown:
        raise ValueError(f"unknown domains {unknown}; available {list(DOMAINS)}")
    if not domains:
        raise ValueError("domains must not be empty")
    turn_counts = list(turn_counts or [3, 4, 5, 6])
    if not turn_counts or any(isinstance(v, bool) or int(v) not in (3, 4, 5, 6) for v in turn_counts):
        raise ValueError("turn_counts must contain only integers from 3 through 6")
    turn_counts = [int(v) for v in turn_counts]
    stats = stats if stats is not None else {}
    stats.setdefault("rejected", {})
    stats.setdefault("emitted", 0)

    specs = [(domains[i % len(domains)], turn_counts[(i // len(domains)) % len(turn_counts)]) for i in range(n)]
    random.Random(f"{seed}:layout").shuffle(specs)
    fault_flags = _quota_flags(n, fault_prob, seed, "fault")
    schema_flags = _quota_flags(n, schema_prob, seed, "schema")
    writes_by_domain = {"filesystem": FS_WRITES, "trading": TRADE_WRITES, "retail": RETAIL_WRITES}

    for index, (domain, turn_count) in enumerate(specs):
        rejected = 0
        while True:
            rng = random.Random(f"{seed}:{index}:{rejected}")
            if domain == "filesystem":
                initial_state, stages, scenario = _filesystem_dialogue(rng, turn_count)
                final_answer = None
            elif domain == "trading":
                initial_state, stages, scenario, final_answer = _trading_dialogue(rng, turn_count)
            else:
                initial_state, stages, scenario = _retail_dialogue(rng, turn_count)
                final_answer = None
            injected = _inject_fault(stages, rng, writes_by_domain[domain]) if index in fault_flags else None
            fault, retry_at = injected if injected is not None else (None, None)
            task_id = f"{SOURCE}-{domain}-{scenario}-{seed}-{index}"
            task = _make_task(
                task_id=task_id,
                domain=domain,
                scenario=scenario,
                initial_state=initial_state,
                stages=stages,
                fault=fault,
                retry_at=retry_at,
                slack=slack,
                final_answer=final_answer,
            )
            task = with_tool_distractors(task, style_index=index)
            task = apply_schema_style(task, style_index=index) if index in schema_flags else mark_canonical(task)
            if verify:
                try:
                    verify_task(task)
                except ValueError as exc:
                    rejected += 1
                    reason = str(exc).split(" (")[0][:80]
                    stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                    if rejected >= 12:
                        raise RuntimeError(f"too many rejected tasks for {task_id}: {stats['rejected']}") from exc
                    continue
            stats["emitted"] += 1
            yield task
            break


__all__ = [
    "DOMAINS",
    "LICENSE",
    "MAX_TURNS_CAP",
    "SOURCE",
    "SYSTEM_PROMPT",
    "generate_tasks",
    "verify_task",
]
