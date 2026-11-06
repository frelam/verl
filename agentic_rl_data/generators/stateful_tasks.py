"""Procedural long-horizon ``mt_fc`` tasks for the stateful ``mock_fc`` env.

Why procedural (AgentGym / tau-bench / BFCL-v3 style, but generated locally)
----------------------------------------------------------------------------
Replay-based multi-turn data (FunReason-MT / APIGen-MT / Toucan) can only
check "did the policy reproduce the reference calls", which punishes valid
alternative plans and cannot teach recovery (the first deviation ends the
episode).  A stateful simulator scores the *outcome* instead, so planning,
exploration, retries and self-correction are all free to happen.  The
generator builds the initial state, the instruction, the goal assertions and
one ground-truth plan together, then **replays the plan through the real env
and reward before yielding the task** (:func:`verify_task`):

* the GT plan must reach ``goal_satisfied=True`` and reward ``1.0``;
* a do-nothing trajectory must fail (the goal is never trivially true);
* with fault injection, dropping the retry must fail (the fault really
  blocks the goal, so "recovery" is what gets rewarded).

Anti-hacking properties of the goals
------------------------------------
* every goal checks the target **and** that non-target objects are
  unchanged (decoy files keep their content, other orders keep their status,
  balances are exact), so "touch everything" / "delete everything" fails;
* information the policy needs (order ids, file locations, prices) only
  exists inside tool results, so the policy has to read before it writes;
* requested values are reported in ``<answer>...</answer>`` and checked
  strictly (:func:`envs.mock_fc.answer_tag_matches`), so listing candidates
  does not pay.

Difficulty = plan length: easy 3-5 GT calls, medium 6-9, hard 10-13; the
turn budget is ``len(plan) + 1 (final reply) + slack`` capped at 15.
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..envs.base import build_env
from ..envs.mock_fc import DEFAULT_FAULT_MESSAGE
from ..rewards import dispatch_reward
from ..schema import validate_task

SOURCE = "procedural_mt"
LICENSE = "generated (repo-internal, no third-party data)"
MAX_TURNS_CAP = 15

SYSTEM_PROMPT = (
    "You are an autonomous agent that completes the user's request by calling the provided tools.\n"
    "Work step by step: plan what you need to know, call tools to look it up, check every tool result, "
    "and only then make changes. Tool calls can fail; when a result starts with 'Error', read it, fix "
    "the cause (or retry if it is a temporary failure) and continue. Never guess ids, paths or values "
    "that you have not seen in a tool result or in the request.\n"
    "When the task is fully done, reply to the user without calling any tool. If the user asks for a "
    "value, put exactly that value inside <answer></answer> in your final reply."
)

_FIRST = ["alice", "bruno", "chen", "dana", "emeka", "farah", "gita", "hiro", "ines", "jonas",
          "kira", "liam", "mona", "nils", "omar", "priya", "quinn", "rosa", "sven", "tara"]
_LAST = ["smith", "garcia", "wang", "kumar", "novak", "silva", "berg", "okafor", "rossi", "tanaka"]
_STREETS = ["Maple Ave", "Oak St", "Pine Rd", "Cedar Ln", "Elm St", "Birch Way", "Lake Dr", "Hill Rd"]
_CITIES = ["Austin", "Denver", "Boston", "Seattle", "Chicago", "Portland", "Phoenix", "Atlanta"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _s(desc: str) -> dict[str, str]:
    return {"type": "string", "description": desc}


def _c(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


@dataclass
class GeneratedTask:
    """Generator output before it is wrapped into the unified task schema."""

    domain: str
    template: str
    user: str
    tools: list[dict[str, Any]]
    initial_state: dict[str, Any]
    goal: list[dict[str, Any]]
    plan: list[list[dict[str, Any]]]
    final_answer: str | None = None
    faults: list[dict[str, Any]] = field(default_factory=list)
    #: index of the plan group that is the retry after an injected fault
    retry_groups: list[int] = field(default_factory=list)


def _inject_fault(rng: random.Random, task: GeneratedTask, write_tools: set[str], prob: float) -> None:
    """Make the first call to one write tool fail transiently; add the retry to the plan."""
    if rng.random() >= prob:
        return
    candidates = [
        index
        for index, group in enumerate(task.plan)
        if len(group) == 1 and group[0]["name"] in write_tools
    ]
    if not candidates:
        return
    index = rng.choice(candidates)
    tool = task.plan[index][0]["name"]
    # only the *first* call to this tool fails, so it must be its first use
    if any(call["name"] == tool for group in task.plan[:index] for call in group):
        return
    task.faults.append({"tool": tool, "count": 1, "message": DEFAULT_FAULT_MESSAGE})
    task.plan.insert(index + 1, copy.deepcopy(task.plan[index]))
    task.retry_groups.append(index + 1)


#: probability that independent reads are issued as one parallel group
PARALLEL_READ_PROB = 0.35


def _reads(rng: random.Random, calls: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Independent reads: one parallel group (teaches parallel calls) or sequential."""
    if len(calls) > 1 and rng.random() < PARALLEL_READ_PROB:
        return [calls]
    return [[call] for call in calls]


# ---------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------

FS_TOOLS = [
    _tool("list_dir", "List the entries of a directory.", {"path": _s("Absolute directory path.")}, ["path"]),
    _tool("read_file", "Read the full content of a file.", {"path": _s("Absolute file path.")}, ["path"]),
    _tool("write_file", "Create or overwrite a file with the given content (parent directory must exist).",
          {"path": _s("Absolute file path."), "content": _s("Full new content.")}, ["path", "content"]),
    _tool("mkdir", "Create one directory. The parent directory must already exist.",
          {"path": _s("Absolute directory path.")}, ["path"]),
    _tool("move_file", "Move a file to a new path (destination must not exist).",
          {"source": _s("Existing file path."), "destination": _s("New file path.")}, ["source", "destination"]),
    _tool("copy_file", "Copy a file to a new path (destination must not exist).",
          {"source": _s("Existing file path."), "destination": _s("New file path.")}, ["source", "destination"]),
    _tool("delete_file", "Delete a file.", {"path": _s("Absolute file path.")}, ["path"]),
    _tool("delete_dir", "Delete an empty directory.", {"path": _s("Absolute directory path.")}, ["path"]),
]
FS_WRITES = {"write_file", "mkdir", "move_file", "copy_file", "delete_file"}

_FS_TOPICS = ["budget", "roadmap", "standup", "retro", "hiring", "offsite", "launch", "audit", "pricing", "vendor"]


def _fs_unchanged(files: dict[str, str], keep: list[str]) -> list[dict[str, Any]]:
    return [
        {"check": "file_content", "path": path, "expected": files[path], "name": f"unchanged:{path}"}
        for path in keep
    ]


def fs_archive_by_keyword(rng: random.Random, level: int) -> GeneratedTask:
    n_files = {0: 3, 1: 4, 2: 6}[level]
    topics = rng.sample(_FS_TOPICS, n_files)
    invoice = rng.randint(1000, 9999)
    target_index = rng.randrange(n_files)
    files: dict[str, str] = {}
    paths = []
    for index, topic in enumerate(topics):
        path = f"/inbox/{topic}_{rng.randint(10, 99)}.txt"
        if index == target_index:
            body = f"Notes on {topic}.\nPayment reference: invoice #{invoice}, due next month."
        else:
            decoy = invoice + rng.choice([-7, 3, 11, 101])
            body = f"Notes on {topic}.\nPayment reference: invoice #{decoy}, already settled."
        files[path] = body
        paths.append(path)
    year = rng.choice([2023, 2024, 2025])
    archive_exists = level == 0
    dirs = ["/", "/inbox"] + (["/archive"] if archive_exists else [])
    target = paths[target_index]
    name = target.rsplit("/", 1)[1]
    destination = f"/archive/{year}/{name}"
    plan: list[list[dict[str, Any]]] = [[_c("list_dir", path="/inbox")]]
    plan += _reads(rng, [_c("read_file", path=path) for path in paths])
    if not archive_exists:
        plan.append([_c("mkdir", path="/archive")])
    plan.append([_c("mkdir", path=f"/archive/{year}")])
    plan.append([_c("move_file", source=target, destination=destination)])
    others = [path for path in paths if path != target]
    goal = [
        {"check": "file_content", "path": destination, "expected": files[target], "name": "archived"},
        {"check": "file_not_exists", "path": target, "name": "removed_from_inbox"},
        {"check": "entries", "path": "/inbox", "mode": "length", "expected": n_files - 1, "name": "inbox_count"},
        {"check": "entries", "path": f"/archive/{year}", "mode": "length", "expected": 1, "name": "archive_count"},
    ] + _fs_unchanged(files, others)
    user = (
        f"One of the files in /inbox mentions invoice #{invoice} as still due. Move that file into "
        f"/archive/{year}/ keeping its file name. Leave every other file untouched."
    )
    return GeneratedTask("filesystem", "archive_by_keyword", user, FS_TOOLS,
                         {"files": files, "dirs": dirs}, goal, plan)


def fs_backup_and_edit(rng: random.Random, level: int) -> GeneratedTask:
    app = rng.choice(["billing", "search", "gateway", "worker", "auth"])
    old_port, new_port = rng.choice([(8080, 9090), (5000, 5050), (3000, 3100), (8000, 8443)])
    keys = {"host": "0.0.0.0", "port": str(old_port), "workers": str(rng.randint(2, 16)),
            "log_level": rng.choice(["info", "debug", "warning"])}
    original = "[server]\n" + "\n".join(f"{k} = {v}" for k, v in keys.items()) + "\n"
    edited = original.replace(f"port = {old_port}", f"port = {new_port}")
    path = f"/etc/{app}/config.ini"
    files = {path: original}
    n_extra = {0: 0, 1: 1, 2: 2}[level]
    extra_edits = []
    if n_extra:
        # level 1/2: also lower log level / bump workers, in the same file
        if n_extra >= 1:
            edited = edited.replace(f"log_level = {keys['log_level']}", "log_level = error")
            extra_edits.append("set log_level to error")
        if n_extra >= 2:
            edited = edited.replace(f"workers = {keys['workers']}", f"workers = {int(keys['workers']) * 2}")
            extra_edits.append("double the number of workers")
    other = f"/etc/{app}/secrets.ini"
    files[other] = f"[secrets]\ntoken = {hashlib.sha1(app.encode()).hexdigest()[:16]}\n"
    backup = path + ".bak"
    plan = [
        [_c("list_dir", path=f"/etc/{app}")],
        [_c("read_file", path=path)],
        [_c("copy_file", source=path, destination=backup)],
        [_c("write_file", path=path, content=edited)],
    ]
    changes = [f"change port from {old_port} to {new_port}"] + extra_edits
    user = (
        f"In {path}: first save an exact backup copy as {backup}, then " + ", ".join(changes)
        + ". Keep every other line and the formatting exactly as they are."
    )
    goal = [
        {"check": "file_content", "path": backup, "expected": original, "name": "backup"},
        {"check": "file_content", "path": path, "expected": edited, "name": "edited"},
    ] + _fs_unchanged(files, [other])
    return GeneratedTask("filesystem", "backup_and_edit", user, FS_TOOLS,
                         {"files": files, "dirs": ["/", "/etc", f"/etc/{app}"]}, goal, plan)


def fs_cleanup_and_count(rng: random.Random, level: int) -> GeneratedTask:
    subdirs = {0: ["/var/log/app"], 1: ["/var/log/app", "/var/log/app/old"],
               2: ["/var/log/app", "/var/log/app/old", "/var/log/app/archive"]}[level]
    per_dir = {0: 3, 1: 2, 2: 2}[level]
    files: dict[str, str] = {}
    empty: list[str] = []
    for directory in subdirs:
        for index in range(per_dir):
            path = f"{directory}/run_{rng.randint(100, 999)}_{index}.log"
            if rng.random() < 0.45:
                files[path] = ""
                empty.append(path)
            else:
                files[path] = f"started\njob finished in {rng.randint(1, 90)}s\n"
    if not empty:  # at least one empty file
        path = next(iter(files))
        files[path] = ""
        empty.append(path)
    plan: list[list[dict[str, Any]]] = []
    for directory in subdirs:
        plan.append([_c("list_dir", path=directory)])
        plan += _reads(rng, [_c("read_file", path=p) for p in files if p.rsplit("/", 1)[0] == directory])
    plan += [[_c("delete_file", path=path)] for path in empty]
    keep = [path for path in files if path not in empty]
    goal = [{"check": "file_not_exists", "path": path, "name": f"deleted:{path}"} for path in empty]
    goal += _fs_unchanged(files, keep)
    where = " and its subdirectories" if len(subdirs) > 1 else ""
    user = (
        f"Delete every empty .log file in /var/log/app{where} (a file is empty when its content is "
        "an empty string). Do not touch non-empty files. Then tell me how many files you deleted."
    )
    dirs = ["/", "/var", "/var/log"] + subdirs
    return GeneratedTask("filesystem", "cleanup_and_count", user, FS_TOOLS,
                         {"files": files, "dirs": dirs}, goal, plan, final_answer=str(len(empty)))


# ---------------------------------------------------------------------------
# trading
# ---------------------------------------------------------------------------

TRADE_TOOLS = [
    _tool("get_balance", "Return the cash balance of the account.", {}, []),
    _tool("get_holdings", "Return the number of shares held per symbol.", {}, []),
    _tool("get_price", "Return the current price of one stock symbol.", {"symbol": _s("Ticker symbol.")}, ["symbol"]),
    _tool("buy", "Buy whole shares at the current price (needs enough cash).",
          {"symbol": _s("Ticker symbol."), "quantity": {"type": "integer", "description": "Number of shares."}},
          ["symbol", "quantity"]),
    _tool("sell", "Sell whole shares at the current price.",
          {"symbol": _s("Ticker symbol."), "quantity": {"type": "integer", "description": "Number of shares."}},
          ["symbol", "quantity"]),
    _tool("deposit", "Deposit cash into the account.",
          {"amount": {"type": "number", "description": "Positive amount in dollars."}}, ["amount"]),
    _tool("transfer", "Transfer cash to another person.",
          {"recipient": _s("Recipient name."), "amount": {"type": "number", "description": "Amount in dollars."}},
          ["recipient", "amount"]),
    _tool("get_history", "Return the most recent account transactions.", {}, []),
]
TRADE_WRITES = {"buy", "sell", "deposit", "transfer"}
_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "ORCL", "INTC", "AMD", "IBM", "SAP", "ADBE"]


def _prices(rng: random.Random, symbols: list[str]) -> dict[str, float]:
    return {symbol: round(rng.uniform(20, 480), 2) for symbol in symbols}


def _trade_goal(holdings: dict[str, int], balance: float) -> list[dict[str, Any]]:
    goal = [{"check": "value", "path": "balance", "mode": "approx", "expected": round(balance, 2), "name": "balance"}]
    goal += [
        {"check": "value", "path": f"holdings.{symbol}", "expected": qty, "name": f"holding:{symbol}"}
        for symbol, qty in sorted(holdings.items())
    ]
    return goal


def trade_rotate(rng: random.Random, level: int) -> GeneratedTask:
    n_sell = {0: 1, 1: 2, 2: 3}[level]
    symbols = rng.sample(_SYMBOLS, n_sell + 2)
    sells, buy_symbol, keep_symbol = symbols[:n_sell], symbols[n_sell], symbols[n_sell + 1]
    prices = _prices(rng, symbols)
    holdings = {symbol: rng.randint(3, 40) for symbol in sells}
    holdings[keep_symbol] = rng.randint(1, 20)
    balance = round(rng.uniform(50, 900), 2)
    plan: list[list[dict[str, Any]]] = [[_c("get_holdings")]]
    plan += _reads(rng, [_c("get_price", symbol=s) for s in sells + [buy_symbol]])
    cash = balance
    final_holdings = dict(holdings)
    for symbol in sells:
        plan.append([_c("sell", symbol=symbol, quantity=holdings[symbol])])
        cash += prices[symbol] * holdings[symbol]
        final_holdings[symbol] = 0
    plan.append([_c("get_balance")])
    quantity = int((cash + 1e-9) // prices[buy_symbol])
    if quantity < 1:
        cash += 1000.0
        plan.insert(-1, [_c("deposit", amount=1000)])
        quantity = int((cash + 1e-9) // prices[buy_symbol])
    plan.append([_c("buy", symbol=buy_symbol, quantity=quantity)])
    cash -= prices[buy_symbol] * quantity
    final_holdings[buy_symbol] = quantity
    sold = " and ".join(sells)
    user = (
        f"Sell all of my {sold} shares. Then use all available cash to buy as many whole shares of "
        f"{buy_symbol} as possible. Do not touch my {keep_symbol} position. Finally tell me the cash "
        "balance left, rounded to 2 decimals."
    )
    deposit_needed = any(call["name"] == "deposit" for group in plan for call in group)
    if deposit_needed:
        user = user.replace("Then use", "Then deposit $1000 and use")
    goal = _trade_goal(final_holdings, cash)
    return GeneratedTask("trading", "rotate", user, TRADE_TOOLS,
                         {"balance": balance, "holdings": holdings, "prices": prices, "history": []},
                         goal, plan, final_answer=f"{cash:.2f}")


def trade_conditional(rng: random.Random, level: int) -> GeneratedTask:
    n_checks = {0: 1, 1: 2, 2: 3}[level]
    symbols = rng.sample(_SYMBOLS, n_checks * 2)
    prices = _prices(rng, symbols)
    balance = round(rng.uniform(20000, 60000), 2)
    plan: list[list[dict[str, Any]]] = []
    lines = []
    holdings: dict[str, int] = {}
    cash = balance
    for index in range(n_checks):
        a, b = symbols[2 * index], symbols[2 * index + 1]
        threshold = round(prices[a] + rng.choice([-25, -10, 10, 25]), 0)
        qty_a, qty_b = rng.randint(2, 12), rng.randint(2, 12)
        lines.append(
            f"- if {a} trades below ${threshold:.0f}, buy {qty_a} {a}; otherwise buy {qty_b} {b}."
        )
        plan.append([_c("get_price", symbol=a)])
        if prices[a] < threshold:
            chosen, qty = a, qty_a
        else:
            chosen, qty = b, qty_b
            plan.append([_c("get_price", symbol=b)])
        plan.append([_c("buy", symbol=chosen, quantity=qty)])
        holdings[chosen] = holdings.get(chosen, 0) + qty
        cash -= prices[chosen] * qty
    plan.append([_c("get_balance")])
    user = "Please run these orders in order:\n" + "\n".join(lines) + "\nThen tell me my remaining cash (2 decimals)."
    goal = _trade_goal(holdings, cash)
    goal += [{"check": "absent", "path": f"holdings.{s}", "name": f"untouched:{s}"} for s in symbols if s not in holdings]
    return GeneratedTask("trading", "conditional", user, TRADE_TOOLS,
                         {"balance": balance, "holdings": {}, "prices": prices, "history": []},
                         goal, plan, final_answer=f"{cash:.2f}")


def trade_pay_with_shortfall(rng: random.Random, level: int) -> GeneratedTask:
    n_payees = {0: 1, 1: 2, 2: 3}[level]
    payees = [name.title() for name in rng.sample(_FIRST, n_payees)]
    amounts = [rng.randint(80, 900) for _ in payees]
    balance = round(sum(amounts) * rng.uniform(0.2, 0.8), 2)
    shortfall = round(sum(amounts) - balance, 2)
    plan = [[_c("get_balance")], [_c("deposit", amount=shortfall)]]
    plan += [[_c("transfer", recipient=p, amount=a)] for p, a in zip(payees, amounts)]
    pays = ", ".join(f"${a} to {p}" for p, a in zip(payees, amounts))
    user = (
        f"I need to pay {pays}. If my cash is not enough for all of them, first deposit exactly the "
        "missing amount (no more), then make the transfers."
    )
    goal = [
        {"check": "value", "path": "balance", "mode": "approx", "expected": 0.0, "name": "balance_zero"},
        {"check": "count", "path": "history", "expected": 1 + n_payees, "name": "n_transactions"},
    ]
    # every payee received exactly their amount (transfer order is free)
    for payee, amount in zip(payees, amounts):
        goal.append({
            "check": "any_of",
            "name": f"paid:{payee}",
            "checks": [
                {"check": "all_of", "checks": [
                    {"check": "value", "path": f"history.{k}.recipient", "expected": payee},
                    {"check": "value", "path": f"history.{k}.amount", "mode": "approx", "expected": float(amount)},
                ]}
                for k in range(1 + n_payees)
            ],
        })
    return GeneratedTask("trading", "pay_with_shortfall", user, TRADE_TOOLS,
                         {"balance": balance, "holdings": {}, "prices": {}, "history": []}, goal, plan)


# ---------------------------------------------------------------------------
# retail (tau-bench style)
# ---------------------------------------------------------------------------

RETAIL_TOOLS = [
    _tool("find_user_by_email", "Look up a customer's user id by email.", {"email": _s("Customer email.")}, ["email"]),
    _tool("get_user_details", "Return a customer's profile, address and order ids.", {"user_id": _s("User id.")}, ["user_id"]),
    _tool("get_order_details", "Return the status, items and address of one order.", {"order_id": _s("Order id, e.g. '#W1234567'.")}, ["order_id"]),
    _tool("get_product_details", "Return a product and all its variants (item ids, options, availability).",
          {"product_id": _s("Product id.")}, ["product_id"]),
    _tool("cancel_pending_order", "Cancel a pending order.",
          {"order_id": _s("Order id."), "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]}},
          ["order_id", "reason"]),
    _tool("modify_order_address", "Change the shipping address of a pending order.",
          {"order_id": _s("Order id."), "address": _s("Full new address.")}, ["order_id", "address"]),
    _tool("return_delivered_items", "Request a return for some items of a delivered order; returns the refund.",
          {"order_id": _s("Order id."), "item_ids": {"type": "array", "items": {"type": "string"}}}, ["order_id", "item_ids"]),
    _tool("exchange_delivered_items", "Exchange items of a delivered order for other variants of the same product.",
          {"order_id": _s("Order id."), "item_ids": {"type": "array", "items": {"type": "string"}},
           "new_item_ids": {"type": "array", "items": {"type": "string"}}}, ["order_id", "item_ids", "new_item_ids"]),
]
RETAIL_WRITES = {"cancel_pending_order", "modify_order_address", "return_delivered_items", "exchange_delivered_items"}

_PRODUCTS = {
    "lamp": ["white", "black", "brass"], "backpack": ["green", "navy", "grey"],
    "headphones": ["black", "silver", "blue"], "kettle": ["steel", "red", "cream"],
    "jacket": ["olive", "black", "tan"], "keyboard": ["white", "black", "pink"],
    "mug": ["blue", "white", "yellow"], "sneakers": ["red", "white", "black"],
}


def _retail_world(rng: random.Random, n_orders: int, statuses: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    user_id = f"{first}_{last}_{rng.randint(1000, 9999)}"
    email = f"{first}.{last}{rng.randint(1, 99)}@example.com"
    address = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    products: dict[str, Any] = {}
    for index, (product, colors) in enumerate(sorted(_PRODUCTS.items())):
        pid = f"P{1000 + index}"
        variants = {}
        for color in colors:
            item_id = str(rng.randint(10**9, 10**10 - 1))
            variants[item_id] = {"options": {"color": color}, "price": round(rng.uniform(15, 250), 2),
                                 "available": rng.random() < 0.8}
        products[pid] = {"name": product, "variants": variants}
    orders: dict[str, Any] = {}
    product_ids = rng.sample(sorted(products), n_orders)
    for index in range(n_orders):
        order_id = f"#W{rng.randint(10**6, 10**7 - 1)}"
        pid = product_ids[index]
        item_id, variant = rng.choice(sorted(products[pid]["variants"].items()))
        orders[order_id] = {
            "user_id": user_id, "status": statuses[index], "address": address,
            "items": [{"item_id": item_id, "product_id": pid, "name": products[pid]["name"],
                       "options": dict(variant["options"]), "price": variant["price"]}],
        }
    users = {user_id: {"name": f"{first.title()} {last.title()}", "email": email,
                       "address": address, "orders": list(orders)}}
    meta = {"user_id": user_id, "email": email, "address": address, "name": users[user_id]["name"]}
    return {"users": users, "orders": orders, "products": products}, meta


def _orders_unchanged(state: dict[str, Any], skip: set[str]) -> list[dict[str, Any]]:
    goal = []
    for order_id, order in state["orders"].items():
        if order_id in skip:
            continue
        goal.append({"check": "value", "path": f"orders.{order_id}.status", "expected": order["status"],
                     "name": f"unchanged_status:{order_id}"})
        goal.append({"check": "value", "path": f"orders.{order_id}.address", "expected": order["address"],
                     "name": f"unchanged_address:{order_id}"})
    return goal


def _retail_lookup(meta: dict[str, Any], state: dict[str, Any], rng: random.Random) -> list[list[dict[str, Any]]]:
    plan = [[_c("find_user_by_email", email=meta["email"])], [_c("get_user_details", user_id=meta["user_id"])]]
    plan += _reads(rng, [_c("get_order_details", order_id=o) for o in state["orders"]])
    return plan


def retail_cancel(rng: random.Random, level: int) -> GeneratedTask:
    n_orders = {0: 2, 1: 3, 2: 5}[level]
    statuses = ["pending"] * n_orders
    statuses[rng.randrange(n_orders)] = "delivered"
    if n_orders >= 3:
        statuses[rng.randrange(n_orders)] = "delivered"
    if "pending" not in statuses:
        statuses[0] = "pending"
    state, meta = _retail_world(rng, n_orders, statuses)
    pending = [o for o, v in state["orders"].items() if v["status"] == "pending"]
    target = rng.choice(pending)
    item = state["orders"][target]["items"][0]
    reason = rng.choice(["no longer needed", "ordered by mistake"])
    plan = _retail_lookup(meta, state, rng)
    plan.append([_c("cancel_pending_order", order_id=target, reason=reason)])
    user = (
        f"Hi, I'm {meta['name']} ({meta['email']}). Please cancel my pending order with the "
        f"{item['options']['color']} {item['name']} - the reason is '{reason}'. Keep my other orders as they are."
    )
    goal = [
        {"check": "value", "path": f"orders.{target}.status", "expected": "cancelled", "name": "cancelled"},
        {"check": "value", "path": f"orders.{target}.cancel_reason", "expected": reason, "name": "reason"},
    ] + _orders_unchanged(state, {target})
    return GeneratedTask("retail", "cancel", user, RETAIL_TOOLS, state, goal, plan)


def retail_exchange(rng: random.Random, level: int) -> GeneratedTask:
    n_orders = {0: 1, 1: 2, 2: 4}[level]
    for _ in range(50):
        statuses = ["delivered"] + [rng.choice(["pending", "delivered"]) for _ in range(n_orders - 1)]
        state, meta = _retail_world(rng, n_orders, statuses)
        target = next(iter(state["orders"]))
        item = state["orders"][target]["items"][0]
        variants = state["products"][item["product_id"]]["variants"]
        options = [(iid, v) for iid, v in sorted(variants.items()) if iid != item["item_id"] and v["available"]]
        if options:
            break
    else:  # pragma: no cover - probability ~0
        raise RuntimeError("could not build an exchangeable world")
    new_id, new_variant = rng.choice(options)
    plan = _retail_lookup(meta, state, rng)
    plan.append([_c("get_product_details", product_id=item["product_id"])])
    plan.append([_c("exchange_delivered_items", order_id=target, item_ids=[item["item_id"]], new_item_ids=[new_id])])
    user = (
        f"I'm {meta['name']}, email {meta['email']}. In my delivered order I got the "
        f"{item['options']['color']} {item['name']}; I'd like to exchange it for the "
        f"{new_variant['options']['color']} one."
    )
    goal = [
        {"check": "value", "path": f"orders.{target}.status", "expected": "exchange requested", "name": "exchanged"},
        {"check": "value", "path": f"orders.{target}.exchange.{item['item_id']}", "expected": new_id, "name": "new_item"},
    ] + _orders_unchanged(state, {target})
    return GeneratedTask("retail", "exchange", user, RETAIL_TOOLS, state, goal, plan)


def retail_return_and_refund(rng: random.Random, level: int) -> GeneratedTask:
    n_orders = {0: 2, 1: 3, 2: 4}[level]
    statuses = ["delivered"] * n_orders
    statuses[-1] = "pending"
    state, meta = _retail_world(rng, n_orders, statuses)
    delivered = [o for o, v in state["orders"].items() if v["status"] == "delivered"]
    n_targets = 1 if level < 2 else 2
    targets = rng.sample(delivered, min(n_targets, len(delivered)))
    plan = _retail_lookup(meta, state, rng)
    refund = 0.0
    names = []
    for order_id in targets:
        item = state["orders"][order_id]["items"][0]
        names.append(f"the {item['options']['color']} {item['name']}")
        refund += item["price"]
        plan.append([_c("return_delivered_items", order_id=order_id, item_ids=[item["item_id"]])])
    user = (
        f"Hello, {meta['name']} here ({meta['email']}). I want to return {' and '.join(names)} from my "
        "delivered orders. How much will I be refunded in total?"
    )
    goal = [
        {"check": "value", "path": f"orders.{o}.status", "expected": "return requested", "name": f"returned:{o}"}
        for o in targets
    ] + _orders_unchanged(state, set(targets))
    return GeneratedTask("retail", "return_and_refund", user, RETAIL_TOOLS, state, goal, plan,
                         final_answer=f"{refund:.2f}")


def retail_move_pending(rng: random.Random, level: int) -> GeneratedTask:
    n_orders = {0: 2, 1: 4, 2: 6}[level]
    statuses = [rng.choice(["pending", "pending", "delivered"]) for _ in range(n_orders)]
    statuses[0] = "pending"
    state, meta = _retail_world(rng, n_orders, statuses)
    new_address = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    while new_address == meta["address"]:
        new_address = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    pending = [o for o, v in state["orders"].items() if v["status"] == "pending"]
    plan = _retail_lookup(meta, state, rng)
    plan += [[_c("modify_order_address", order_id=o, address=new_address)] for o in pending]
    user = (
        f"I'm {meta['name']} ({meta['email']}) and I'm moving. Ship every order that has not been "
        f"delivered yet to my new address: {new_address}. Then tell me how many orders you updated."
    )
    goal = [
        {"check": "value", "path": f"orders.{o}.address", "expected": new_address, "name": f"moved:{o}"}
        for o in pending
    ] + _orders_unchanged(state, set(pending))
    return GeneratedTask("retail", "move_pending", user, RETAIL_TOOLS, state, goal, plan,
                         final_answer=str(len(pending)))


def retail_multi_request(rng: random.Random, level: int) -> GeneratedTask:
    """Composite request (cancel + return + address change + refund report): 10-15 calls."""
    n_orders = {0: 3, 1: 4, 2: 5}[level]
    statuses = ["pending", "delivered", "pending"] + [rng.choice(["pending", "delivered"]) for _ in range(n_orders - 3)]
    rng.shuffle(statuses)
    state, meta = _retail_world(rng, n_orders, statuses)
    orders = state["orders"]
    pending = [o for o, v in orders.items() if v["status"] == "pending"]
    delivered = [o for o, v in orders.items() if v["status"] == "delivered"]
    cancel = rng.choice(pending)
    ret = rng.choice(delivered)
    move = [o for o in pending if o != cancel]
    new_address = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    while new_address == meta["address"]:
        new_address = f"{rng.randint(10, 999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    cancel_item, ret_item = orders[cancel]["items"][0], orders[ret]["items"][0]
    reason = rng.choice(["no longer needed", "ordered by mistake"])
    plan = _retail_lookup(meta, state, rng)
    plan.append([_c("cancel_pending_order", order_id=cancel, reason=reason)])
    plan.append([_c("return_delivered_items", order_id=ret, item_ids=[ret_item["item_id"]])])
    plan += [[_c("modify_order_address", order_id=o, address=new_address)] for o in move]
    user = (
        f"Hi, this is {meta['name']} ({meta['email']}). Three things: (1) cancel the pending order with the "
        f"{cancel_item['options']['color']} {cancel_item['name']} because it is '{reason}'; (2) return the "
        f"{ret_item['options']['color']} {ret_item['name']} I received; (3) send all my other pending orders "
        f"to {new_address}. Finally tell me the refund amount for the return."
    )
    goal = [
        {"check": "value", "path": f"orders.{cancel}.status", "expected": "cancelled", "name": "cancelled"},
        {"check": "value", "path": f"orders.{cancel}.address", "expected": meta["address"], "name": "cancelled_not_moved"},
        {"check": "value", "path": f"orders.{ret}.status", "expected": "return requested", "name": "returned"},
    ]
    goal += [{"check": "value", "path": f"orders.{o}.address", "expected": new_address, "name": f"moved:{o}"} for o in move]
    goal += _orders_unchanged(state, {cancel, ret, *move})
    return GeneratedTask("retail", "multi_request", user, RETAIL_TOOLS, state, goal, plan,
                         final_answer=f"{ret_item['price']:.2f}")


def fs_reorganize(rng: random.Random, level: int) -> GeneratedTask:
    """Sort files into per-extension folders and write an index: 8-13 calls."""
    n_files = {0: 3, 1: 4, 2: 6}[level]
    exts = ["csv", "md", "py"]
    files: dict[str, str] = {}
    names = []
    topics = rng.sample(_FS_TOPICS, n_files)
    for index in range(n_files):
        ext = exts[index] if index < len(exts) else rng.choice(exts)
        name = f"{topics[index]}.{ext}"
        files[f"/project/{name}"] = f"# {name}\nline {rng.randint(1, 99)}\n"
        names.append(name)
    used = sorted({name.rsplit(".", 1)[1] for name in names})
    plan: list[list[dict[str, Any]]] = [[_c("list_dir", path="/project")]]
    plan += [[_c("mkdir", path=f"/project/{ext}")] for ext in used]
    plan += [
        [_c("move_file", source=f"/project/{n}", destination=f"/project/{n.rsplit('.', 1)[1]}/{n}")]
        for n in names
    ]
    index_lines = "\n".join(sorted(f"{n.rsplit('.', 1)[1]}/{n}" for n in names))
    plan.append([_c("write_file", path="/project/INDEX.txt", content=index_lines)])
    user = (
        "Tidy up /project: create one sub-folder per file extension (named after the extension, e.g. "
        "/project/csv), move every file into the folder for its extension, then write /project/INDEX.txt "
        "listing the new relative paths (like 'csv/report.csv'), one per line, sorted alphabetically, "
        "with no blank lines."
    )
    goal = [
        {"check": "file_content", "path": f"/project/{n.rsplit('.', 1)[1]}/{n}",
         "expected": files[f"/project/{n}"], "name": f"moved:{n}"}
        for n in names
    ]
    # trailing newline is tolerated; everything else must match exactly
    goal.append({"check": "file_content", "path": "/project/INDEX.txt", "mode": "regex",
                 "expected": "^" + re.escape(index_lines) + r"\n?$", "name": "index"})
    goal.append({"check": "entries", "path": "/project", "expected": sorted(used + ["INDEX.txt"]), "name": "layout"})
    return GeneratedTask("filesystem", "reorganize", user, FS_TOOLS,
                         {"files": files, "dirs": ["/", "/project"]}, goal, plan)


# ---------------------------------------------------------------------------
# registry, verification, task assembly
# ---------------------------------------------------------------------------

DOMAIN_GENERATORS: dict[str, list[tuple[Callable[[random.Random, int], GeneratedTask], set[str]]]] = {
    "filesystem": [(fs_archive_by_keyword, FS_WRITES), (fs_backup_and_edit, FS_WRITES),
                   (fs_cleanup_and_count, FS_WRITES), (fs_reorganize, FS_WRITES)],
    "trading": [(trade_rotate, TRADE_WRITES), (trade_conditional, TRADE_WRITES),
                (trade_pay_with_shortfall, TRADE_WRITES)],
    "retail": [(retail_cancel, RETAIL_WRITES), (retail_exchange, RETAIL_WRITES),
               (retail_return_and_refund, RETAIL_WRITES), (retail_move_pending, RETAIL_WRITES),
               (retail_multi_request, RETAIL_WRITES)],
}


def _final_reply(answer: str | None) -> str:
    return f"All done. <answer>{answer}</answer>" if answer is not None else "All done."


def _replay(task: dict[str, Any], groups: list[list[dict[str, Any]]], reply: str) -> dict[str, Any]:
    env = build_env(task)
    env.reset(task)
    try:
        for group in groups:
            _, done, _ = env.step({"content": "", "tool_calls": copy.deepcopy(group)})
            if done:
                break
        if not env.done:
            env.step({"content": reply, "tool_calls": []})
        return env.trajectory()
    finally:
        env.close()


def verify_task(task: dict[str, Any], retry_groups: list[int] | None = None) -> dict[str, Any]:
    """Self-check one generated task; raises ``ValueError`` when it is unusable."""
    errors = validate_task(task)
    if errors:
        raise ValueError(f"schema: {errors[:3]}")
    plan = task["reference"]["gt_calls"]
    answer = task["reference"]["final_answer"]
    good = _replay(task, plan, _final_reply(answer))
    state = good["env_state"] or {}
    if state.get("goal_satisfied") is not True:
        failed = [c["name"] for c in state.get("checks", []) if not c.get("ok")]
        raise ValueError(f"GT plan does not satisfy the goal (failed checks: {failed[:5]})")
    score = dispatch_reward(task, good)["score"]
    if score < 1.0 - 1e-9:
        raise ValueError(f"GT plan scores {score:.3f} < 1.0")
    idle = _replay(task, [], _final_reply(answer))
    if (idle["env_state"] or {}).get("goal_satisfied") is True:
        raise ValueError("goal is satisfied by a do-nothing trajectory")
    if answer is not None:
        wrong = _replay(task, plan, "All done.")
        if (wrong["env_state"] or {}).get("goal_satisfied") is True:
            raise ValueError("goal is satisfied without reporting the requested value")
    for index in retry_groups or []:
        no_retry = [g for i, g in enumerate(plan) if i != index]
        trajectory = _replay(task, no_retry, _final_reply(answer))
        if (trajectory["env_state"] or {}).get("goal_satisfied") is True:
            raise ValueError("injected fault does not block the goal")
    return {"score": score, "num_turns": good["num_turns"]}


def to_task(gen: GeneratedTask, *, task_id: str, level: int, slack: int) -> dict[str, Any]:
    expected = len(gen.plan) + 1
    max_turns = min(MAX_TURNS_CAP, expected + slack + len(gen.faults))
    config: dict[str, Any] = {
        "impl": "stateful",
        "domain": gen.domain,
        "initial_state": gen.initial_state,
        "goal": gen.goal,
        "max_turns": max_turns,
    }
    if gen.final_answer is not None:
        config["require_final_answer"] = True
        config["final_answer_mode"] = "answer_tag"
    if gen.faults:
        config["faults"] = gen.faults
    # outcome-verified: score efficiency, not similarity to this one GT plan
    config["process_term"] = "efficiency"
    return {
        "id": task_id,
        "source": SOURCE,
        "task_type": "mt_fc",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": gen.user}],
        "tools": copy.deepcopy(gen.tools),
        "env": {"type": "mock_fc", "config": config},
        "reference": {
            "final_answer": gen.final_answer,
            "final_answers": None,
            "gt_calls": gen.plan,
            "gt_turns": None,
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": {
            "max_turns": max_turns,
            "difficulty": float(level),
            "pass_rate": None,
            "license": LICENSE,
            "expected_turns": expected,
            # returns come from the simulator itself (never placeholders)
            "gt_returns_source": "config",
            "group_id": task_id,
            "domain": gen.domain,
            "template": gen.template,
            "n_faults": len(gen.faults),
            "requires_answer": gen.final_answer is not None,
        },
    }


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    domains: list[str] | None = None,
    level_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
    fault_prob: float = 0.3,
    slack: int = 3,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield ``n`` verified tasks, deterministic for a fixed ``seed``.

    Tasks rejected by :func:`verify_task` are counted in ``stats["rejected"]``
    (by reason) and replaced, so exactly ``n`` tasks come out.
    """
    domains = list(domains or sorted(DOMAIN_GENERATORS))
    unknown = sorted(set(domains) - set(DOMAIN_GENERATORS))
    if unknown:
        raise ValueError(f"unknown domains {unknown}; available {sorted(DOMAIN_GENERATORS)}")
    pool = [(d, fn, writes) for d in domains for fn, writes in DOMAIN_GENERATORS[d]]
    stats = stats if stats is not None else {}
    stats.setdefault("rejected", {})
    stats.setdefault("emitted", 0)
    attempt = 0
    while stats["emitted"] < n:
        attempt += 1
        if attempt > n * 5 + 100:
            raise RuntimeError(f"too many rejected tasks: {stats['rejected']}")
        rng = random.Random(f"{seed}:{attempt}")
        domain, fn, writes = pool[rng.randrange(len(pool))]
        level = rng.choices([0, 1, 2], weights=level_weights)[0]
        gen = fn(rng, level)
        _inject_fault(rng, gen, writes, fault_prob)
        if len(gen.plan) + 1 + len(gen.faults) > MAX_TURNS_CAP:
            stats["rejected"]["too_long"] = stats["rejected"].get("too_long", 0) + 1
            continue
        task_id = f"{SOURCE}-{domain}-{gen.template}-{seed}-{attempt}"
        task = to_task(gen, task_id=task_id, level=level, slack=slack)
        if verify:
            try:
                verify_task(task, gen.retry_groups)
            except ValueError as exc:
                reason = str(exc).split(" (")[0][:60]
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue
        stats["emitted"] += 1
        yield task


__all__ = [
    "DOMAIN_GENERATORS",
    "GeneratedTask",
    "LICENSE",
    "SOURCE",
    "SYSTEM_PROMPT",
    "generate_tasks",
    "to_task",
    "verify_task",
]
