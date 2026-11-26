"""Outcome-verified multi-turn tasks with a required same-turn parallel group."""

from __future__ import annotations

import copy
import random
from typing import Any, Iterator

from ..envs.base import build_env
from ..rewards import dispatch_reward
from .schema_distractors import with_tool_distractors
from .schema_styles import apply_schema_style, mark_canonical
from .stateful_tasks import (
    FS_TOOLS,
    FS_WRITES,
    RETAIL_TOOLS,
    RETAIL_WRITES,
    TRADE_TOOLS,
    TRADE_WRITES,
    GeneratedTask,
    _c,
    _fs_unchanged,
    _inject_fault,
    _prices,
    _retail_world,
    _trade_goal,
    to_task,
    verify_task,
)

SOURCE = "parallel_mt"
LICENSE = "generated (repo-internal, no third-party data)"


def _fs_parallel(rng: random.Random, level: int) -> GeneratedTask:
    topics = rng.sample(["alpha", "beta", "gamma", "delta", "epsilon"], 2 + level)
    decoy_path = "/project/keep.txt"
    files = {decoy_path: "do not change\n"}
    calls = []
    goal = []
    for topic in topics:
        content = f"{topic}-{rng.randint(100, 999)}\n"
        path = f"/project/{topic}.txt"
        calls.append(_c("write_file", path=path, content=content))
        goal.append({"check": "file_content", "path": path, "expected": content, "name": f"written:{topic}"})
    index = "\n".join(sorted(f"{topic}.txt" for topic in topics))
    plan = [calls, [_c("list_dir", path="/project")], [_c("write_file", path="/project/INDEX.txt", content=index)]]
    goal.append({"check": "file_content", "path": "/project/INDEX.txt", "expected": index, "name": "index"})
    goal += _fs_unchanged(files, [decoy_path])
    assignments = ", ".join(
        f"{topic}.txt with exact content {call['arguments']['content']!r}"
        for topic, call in zip(topics, calls)
    )
    user = (
        f"In /project, create these files in one batch: {assignments}. "
        "Then list the directory and write INDEX.txt containing just those new file names, sorted, one per line. "
        "Keep keep.txt unchanged."
    )
    task = GeneratedTask("filesystem", "parallel_write_then_index", user, FS_TOOLS,
                         {"files": files, "dirs": ["/", "/project"]}, goal, plan)
    task.retry_groups = []
    return task


def _retail_parallel(rng: random.Random, level: int) -> GeneratedTask:
    n_orders = 2 + level
    state, meta = _retail_world(rng, n_orders, ["pending"] * n_orders)
    order_ids = list(state["orders"])
    address = f"{rng.randint(10, 999)} Parallel Avenue, {rng.choice(['Austin', 'Denver', 'Boston'])}"
    reads = [_c("get_order_details", order_id=order_id) for order_id in order_ids]
    plan = [
        [_c("find_user_by_email", email=meta["email"])],
        [_c("get_user_details", user_id=meta["user_id"])],
        reads,
    ]
    plan += [[_c("modify_order_address", order_id=order_id, address=address)] for order_id in order_ids]
    goal = [
        {"check": "value", "path": f"orders.{order_id}.address", "expected": address, "name": f"moved:{order_id}"}
        for order_id in order_ids
    ]
    goal += [
        {"check": "value", "path": f"orders.{order_id}.status", "expected": "pending", "name": f"status:{order_id}"}
        for order_id in order_ids
    ]
    user = (
        f"I'm {meta['name']} ({meta['email']}). Look up all my pending orders in one batch, then change "
        f"the shipping address of every one to {address}. Do not cancel any order."
    )
    return GeneratedTask("retail", "parallel_order_reads", user, RETAIL_TOOLS, state, goal, plan)


def _trading_parallel(rng: random.Random, level: int) -> GeneratedTask:
    symbols = rng.sample(["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"], 2 + level)
    prices = _prices(rng, symbols)
    balance = round(rng.uniform(15000, 40000), 2)
    quantities = {symbol: rng.randint(1, 4) for symbol in symbols}
    reads = [_c("get_price", symbol=symbol) for symbol in symbols]
    plan = [reads]
    holdings: dict[str, int] = {}
    cash = balance
    for symbol in symbols:
        quantity = quantities[symbol]
        plan.append([_c("buy", symbol=symbol, quantity=quantity)])
        holdings[symbol] = quantity
        cash -= prices[symbol] * quantity
    plan.append([_c("get_balance")])
    orders = ", ".join(f"{quantities[symbol]} shares of {symbol}" for symbol in symbols)
    user = (
        f"Check the prices of {', '.join(symbols)} in one batch. Then buy {orders}, in that order, "
        "and tell me the final cash balance to 2 decimals."
    )
    return GeneratedTask("trading", "parallel_price_reads", user, TRADE_TOOLS,
                         {"balance": balance, "holdings": {}, "prices": prices, "history": []},
                         _trade_goal(holdings, cash), plan, final_answer=f"{cash:.2f}")


_GENERATORS = [
    ("filesystem", _fs_parallel, FS_WRITES),
    ("retail", _retail_parallel, RETAIL_WRITES),
    ("trading", _trading_parallel, TRADE_WRITES),
]


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


def _verify_parallel(task: dict[str, Any], retry_groups: list[int]) -> None:
    result = verify_task(task, retry_groups)
    if result["score"] < 1.0:
        raise ValueError("GT plan is not full reward")
    groups = task["reference"]["gt_calls"]
    required = task["env"]["config"]["required_parallel"][0]
    for index, group in enumerate(groups):
        if group == required:
            serial = groups[:index] + [[call] for call in group] + groups[index + 1 :]
            trajectory = _replay(task, serial)
            if (trajectory["env_state"] or {}).get("goal_satisfied") is True:
                raise ValueError("serialised parallel group still satisfies the goal")
            return
    raise ValueError("required parallel group is absent from the GT plan")


def generate_tasks(
    n: int,
    *,
    seed: int = 0,
    fault_prob: float = 0.25,
    schema_perturb_prob: float = 0.4,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    stats = stats if stats is not None else {}
    stats.setdefault("rejected", {})
    stats.setdefault("emitted", 0)
    attempt = 0
    while stats["emitted"] < n:
        attempt += 1
        if attempt > n * 8 + 100:
            raise RuntimeError(f"too many rejected tasks: {stats['rejected']}")
        rng = random.Random(f"{seed}:parallel:{attempt}")
        _, factory, writes = _GENERATORS[(attempt - 1) % len(_GENERATORS)]
        level = (attempt - 1) % 3
        gen = factory(rng, level)
        _inject_fault(rng, gen, writes, fault_prob)
        task = to_task(gen, task_id=f"{SOURCE}-{gen.domain}-{seed}-{attempt}", level=level, slack=3)
        task["source"] = SOURCE
        required_index = 0 if gen.domain != "retail" else 2
        task["env"]["config"]["required_parallel"] = [copy.deepcopy(gen.plan[required_index])]
        variant_index = stats["emitted"]
        task = with_tool_distractors(task, style_index=variant_index)
        if (variant_index % 5) / 5 < schema_perturb_prob:
            task = apply_schema_style(task, style_index=variant_index)
            task["env"]["config"]["required_parallel"] = [
                copy.deepcopy(task["reference"]["gt_calls"][required_index])
            ]
        else:
            task = mark_canonical(task)
        task["meta"].update({
            "group_id": task["id"],
            "scenario": gen.template,
            "parallel_group_count": 1,
            "user_turn_count": 1,
        })
        if verify:
            try:
                _verify_parallel(task, gen.retry_groups)
            except ValueError as exc:
                reason = str(exc).split(" (")[0][:60]
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue
        stats["emitted"] += 1
        yield task


__all__ = ["LICENSE", "SOURCE", "generate_tasks"]
