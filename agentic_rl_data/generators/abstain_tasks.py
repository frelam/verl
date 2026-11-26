"""Stateful missing-capability, missing-parameter, and inferable tasks.

These remain ``mt_fc`` tasks: the dialogue environment, rather than the generic
single-turn abstain reward, decides which clarification/refusal is correct and
whether business tools may be called.  ``strict_outcome`` prevents a wrong
behaviour from earning process/format consolation reward.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Iterator, Mapping

from ..rewards import dispatch_reward
from .interleaved_tasks import (
    _Stage,
    _inject_fault,
    _make_task,
    _quota_flags,
    _reference_actions,
    _run_actions,
)
from .interleaved_tasks import verify_task as _verify_dialogue
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
    _orders_unchanged,
    _prices,
    _reads,
    _retail_lookup,
    _retail_world,
    _trade_goal,
)

SOURCE = "stateful_abstain"
LICENSE = "generated (repo-internal, no third-party data)"
KINDS = ("miss_func", "miss_param", "inferable")
DOMAINS = ("filesystem", "trading", "retail")

_REFUSAL = "Sorry, I don't have access to that capability and no available tool fits this request."


def _file(path: str, content: str, name: str) -> dict[str, Any]:
    return {"check": "file_content", "path": path, "expected": content, "name": name}


def _no_file(path: str, name: str) -> dict[str, Any]:
    return {"check": "file_not_exists", "path": path, "name": name}


def _text_stage(kind: str, user: str, goal: list[dict[str, Any]], reply: str, content: str,
                slots: list[Any] | None = None) -> _Stage:
    return _Stage(kind, user, copy.deepcopy(goal), [], invariants=copy.deepcopy(goal),
                  reply=reply, slot_keywords=copy.deepcopy(slots or []), content=content)


def _filesystem(kind: str, rng: random.Random) -> tuple[dict[str, Any], list[_Stage], str]:
    source = f"Release note {rng.randint(100, 999)}: ready for review.\n"
    decoy = "Do not modify this unrelated file.\n"
    state = {
        "files": {"/workspace/source.txt": source, "/workspace/decoy.txt": decoy},
        "dirs": ["/", "/workspace"],
    }
    base = [_file("/workspace/source.txt", source, "source"), _file("/workspace/decoy.txt", decoy, "decoy")]
    if kind == "miss_func":
        stage = _text_stage(
            kind,
            "Read /workspace/source.txt and email its contents to the reviewer. Do not create any local files.",
            base,
            "no_valid_tools",
            _REFUSAL,
        )
        return state, [stage], "filesystem_missing_email"

    target = "/workspace/report.txt"
    if kind == "miss_param":
        content = f"Approved release {rng.randint(1000, 9999)}.\n"
        clarify = _text_stage(
            kind,
            "Create /workspace/report.txt with the content I have in mind. The file must be new; do not guess its content.",
            base + [_no_file(target, "report_absent")],
            "request_info",
            "Could you please provide the content for the file?",
            ["content"],
        )
        goal = base + [_file(target, content, "report")]
        execute = _Stage(
            kind,
            f"The content is exactly {content!r}. Now create the file without changing source.txt or decoy.txt.",
            goal,
            [[_c("write_file", path=target, content=content)]],
            invariants=copy.deepcopy(base),
            must_change=[_file(target, content, "report")],
        )
        return state, [clarify, execute], "filesystem_missing_content"

    target = "/workspace/mirror.txt"
    goal = base + [_file(target, source, "mirror")]
    execute = _Stage(
        kind,
        "Make /workspace/mirror.txt exactly match /workspace/source.txt. You can read the source, so do not ask me for its content. Leave both original files unchanged.",
        goal,
        [[_c("read_file", path="/workspace/source.txt")], [_c("write_file", path=target, content=source)]],
        invariants=copy.deepcopy(base),
        must_change=[_file(target, source, "mirror")],
    )
    return state, [execute], "filesystem_infer_content"


def _trading(kind: str, rng: random.Random) -> tuple[dict[str, Any], list[_Stage], str]:
    symbol, decoy = rng.sample(_SYMBOLS, 2)
    prices = _prices(rng, [symbol, decoy])
    balance = round(rng.uniform(5000, 12000), 2)
    holdings = {symbol: 3} if kind == "inferable" else {symbol: 3, decoy: 2}
    state = {"balance": balance, "holdings": holdings, "prices": prices, "history": []}
    initial_goal = _trade_goal(holdings, balance)
    if kind == "miss_func":
        stage = _text_stage(
            kind,
            "Book me a hotel room for tomorrow night. Do not make any trades or transfers.",
            initial_goal,
            "no_valid_tools",
            _REFUSAL,
        )
        return state, [stage], "trading_missing_travel"

    if kind == "miss_param":
        recipient = rng.choice(["Alice", "Bruno", "Chen", "Dana"])
        amount = float(rng.randint(50, 300))
        clarify = _text_stage(
            kind,
            f"Transfer some money to {recipient}. Do not guess how much I mean and do not change my positions.",
            initial_goal,
            "request_info",
            f"Could you please specify the amount to transfer to {recipient}?",
            ["amount"],
        )
        final_holdings = dict(holdings)
        final_balance = round(balance - amount, 2)
        goal = _trade_goal(final_holdings, final_balance)
        execute = _Stage(
            kind,
            f"The amount is ${amount:.2f}. Make exactly that transfer now and leave every position unchanged.",
            goal,
            [[_c("transfer", recipient=recipient, amount=amount)]],
            invariants=[check for check in initial_goal if check.get("name") != "balance"],
            must_change=[check for check in goal if check.get("name") == "balance"],
        )
        return state, [clarify, execute], "trading_missing_amount"

    final_holdings = dict(holdings)
    final_holdings[symbol] += 1
    final_balance = round(balance - prices[symbol], 2)
    goal = _trade_goal(final_holdings, final_balance)
    execute = _Stage(
        kind,
        "Buy one more share of my only current stock position. Infer its ticker from my holdings; do not ask which symbol I mean.",
        goal,
        [[_c("get_holdings")]] + _reads(rng, [_c("get_price", symbol=symbol)]) + [[_c("buy", symbol=symbol, quantity=1)]],
        invariants=[],
        must_change=[check for check in goal if check.get("name") in {"balance", f"holding:{symbol}"}],
    )
    return state, [execute], "trading_infer_symbol"


def _other_address(rng: random.Random, old: str) -> str:
    value = old
    while value == old:
        value = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    return value


def _item_label(state: Mapping[str, Any], order_id: str) -> str:
    item = state["orders"][order_id]["items"][0]
    return f"{item['options']['color']} {item['name']}"


def _retail(kind: str, rng: random.Random) -> tuple[dict[str, Any], list[_Stage], str]:
    statuses = ["pending", "delivered", "delivered"] if kind == "inferable" else ["pending", "delivered", "pending"]
    state, meta = _retail_world(rng, 3, statuses)
    target = next(order_id for order_id, order in state["orders"].items() if order["status"] == "pending")
    unchanged = _orders_unchanged(state, set())
    if kind == "miss_func":
        stage = _text_stage(
            kind,
            f"I'm {meta['name']} ({meta['email']}). Reset the password on my customer account and leave every order untouched.",
            unchanged,
            "no_valid_tools",
            _REFUSAL,
        )
        return state, [stage], "retail_missing_password_reset"

    if kind == "miss_param":
        reason = "no longer needed"
        clarify = _text_stage(
            kind,
            f"I'm {meta['name']} ({meta['email']}). Cancel my pending order with the {_item_label(state, target)} "
            "for the reason I have in mind. Do not guess the reason or look up orders before I provide it.",
            unchanged,
            "request_info",
            "Could you please specify the cancellation reason?",
            ["reason"],
        )
        goal = [
            {"check": "value", "path": f"orders.{target}.status", "expected": "cancelled", "name": "cancelled"},
            {"check": "value", "path": f"orders.{target}.cancel_reason", "expected": reason, "name": "reason"},
        ] + _orders_unchanged(state, {target})
        plan = _retail_lookup(meta, state, rng)
        plan.append([_c("cancel_pending_order", order_id=target, reason=reason)])
        execute = _Stage(
            kind,
            f"The reason is {reason!r}. Now look up and cancel that order; preserve every other order.",
            goal,
            plan,
            invariants=_orders_unchanged(state, {target}),
            must_change=copy.deepcopy(goal[:2]),
        )
        return state, [clarify, execute], "retail_missing_reason"

    old_address = _other_address(rng, meta["address"])
    state["orders"][target]["address"] = old_address
    unchanged = _orders_unchanged(state, set())
    goal = [
        {"check": "value", "path": f"orders.{target}.address", "expected": meta["address"], "name": "profile_address"},
        {"check": "value", "path": f"orders.{target}.status", "expected": "pending", "name": "still_pending"},
    ] + _orders_unchanged(state, {target})
    plan = _retail_lookup(meta, state, rng)
    plan.append([_c("modify_order_address", order_id=target, address=meta["address"])])
    execute = _Stage(
        kind,
        f"I'm {meta['name']} ({meta['email']}). For my only pending order, use the address already on my customer "
        "profile as the shipping address. Infer both the order and address; do not ask me for them.",
        goal,
        plan,
        invariants=[check for check in unchanged if check.get("path") != f"orders.{target}.address"],
        must_change=copy.deepcopy(goal[:1]),
    )
    return state, [execute], "retail_infer_profile_address"


def _build(kind: str, domain: str, rng: random.Random) -> tuple[dict[str, Any], list[_Stage], str]:
    if domain == "filesystem":
        return _filesystem(kind, rng)
    if domain == "trading":
        return _trading(kind, rng)
    return _retail(kind, rng)


def verify_task(task: dict[str, Any], retry_groups: list[int] | None = None) -> dict[str, Any]:
    """Verify GT/idle/retry plus the class-specific behavioural negative."""
    result = _verify_dialogue(task, retry_groups)
    kind = str(task["meta"].get("scenario"))
    if kind == "miss_func" and task["tools"]:
        call = {"name": task["tools"][0]["name"], "arguments": {}}
        trajectory = _run_actions(task, [([call], "", None)], finish=True)
        if float(dispatch_reward(task, trajectory)["score"]) != 0.0:
            raise ValueError("miss_func accepts a spurious business-tool call")
    elif kind == "miss_param":
        first_call = next((copy.deepcopy(group) for group in task["reference"]["gt_calls"] if group), None)
        if first_call is None:
            raise ValueError("miss_param has no execution stage")
        early = _run_actions(task, [(first_call, "", None)], finish=True)
        if float(dispatch_reward(task, early)["score"]) != 0.0:
            raise ValueError("miss_param accepts execution before clarification")
        vague = _run_actions(task, [([], "Could you please clarify?", None)], finish=True)
        if float(dispatch_reward(task, vague)["score"]) != 0.0:
            raise ValueError("miss_param accepts clarification without its required slot")
    elif kind == "inferable":
        clarify = _run_actions(task, [([], "Could you please clarify the missing details?", None)], finish=True)
        if float(dispatch_reward(task, clarify)["score"]) != 0.0:
            raise ValueError("inferable task accepts an unnecessary clarification")
    return result


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    domains: list[str] | None = None,
    kinds: list[str] | None = None,
    fault_prob: float = 0.25,
    schema_prob: float = 0.75,
    slack: int = 3,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield balanced kinds/domains; about one quarter of all tasks has a fault."""
    if n < 0:
        raise ValueError("n must be non-negative")
    domains = list(domains or DOMAINS)
    kinds = list(kinds or KINDS)
    if not domains or set(domains) - set(DOMAINS):
        raise ValueError(f"domains must be a non-empty subset of {list(DOMAINS)}")
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError(f"kinds must be a non-empty subset of {list(KINDS)}")
    stats = stats if stats is not None else {}
    stats.setdefault("rejected", {})
    stats.setdefault("emitted", 0)

    specs = [(kinds[i % len(kinds)], domains[(i // len(kinds)) % len(domains)]) for i in range(n)]
    random.Random(f"{seed}:abstain:layout").shuffle(specs)
    eligible = [index for index, (kind, _) in enumerate(specs) if kind != "miss_func"]
    fault_flags = _quota_flags(n, fault_prob, seed, "abstain:fault", eligible)
    schema_flags = _quota_flags(n, schema_prob, seed, "abstain:schema")
    writes = {"filesystem": FS_WRITES, "trading": TRADE_WRITES, "retail": RETAIL_WRITES}

    for index, (kind, domain) in enumerate(specs):
        rejected = 0
        while True:
            rng = random.Random(f"{seed}:abstain:{index}:{rejected}")
            initial_state, stages, template = _build(kind, domain, rng)
            injected = _inject_fault(stages, rng, writes[domain]) if index in fault_flags else None
            fault, retry_at = injected if injected is not None else (None, None)
            task_id = f"{SOURCE}-{domain}-{kind}-{seed}-{index}"
            task = _make_task(
                task_id=task_id,
                domain=domain,
                scenario=kind,
                initial_state=initial_state,
                stages=stages,
                fault=fault,
                retry_at=retry_at,
                slack=slack,
            )
            task["source"] = SOURCE
            task["env"]["config"]["strict_outcome"] = True
            task["meta"].update(
                {
                    "scenario": kind,
                    "template": template,
                    "abstention_class": kind,
                    "difficulty": float(KINDS.index(kind)),
                    "license": LICENSE,
                }
            )
            for stage in task["env"]["config"]["dialogue"]["stages"]:
                if stage["reply"] in ("request_info", "no_valid_tools"):
                    stage["allow_tools"] = False
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


__all__ = ["DOMAINS", "KINDS", "LICENSE", "SOURCE", "generate_tasks", "verify_task"]
