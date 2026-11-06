"""``st_fc`` reward - single-turn function calling (plan §5).

``score = 1.0`` iff all three hold, else ``0.0``:

1. every ground-truth call is matched one-to-one by an emitted call;
2. no extra / spurious emitted call exists;
3. the trajectory is structurally format-legal (the format gate).

Calls are order-independent - parallel calls may be permuted (plan §2/§5).
``st_fc`` is single-turn, so all ``gt_calls`` turn groups are flattened into
one matching set.  Argument values use AST-ish equality
(:func:`agentic_rl_data.rewards.base.values_equal`), e.g. ``3`` matches
``3.0`` / ``"3"``, and a literal stringification of a list/dict matches the
structured value; strings compare case- and whitespace-insensitively.

Format gate (``base.format_report``): a call to a tool not declared in
``task["tools"]``, a call emitted when no tools are declared at all, a missing
required parameter, an unknown parameter, a value whose type contradicts the
declared JSON schema, or a call without a matching ``tool_results`` entry zeroes
the score even when the argument values happen to match.  The raw
``<tool_call>`` markup is not visible in the normalised trajectory, so "format"
here is structural (see ``base``).

Optional-parameter convention (review T3-07): matching is strict on the label's
argument keys - every ground-truth argument must be present with an equivalent
value (AST-ish equality), and an emitted call with an *extra* argument key is a
mismatch.  A declared *optional* JSON-schema parameter may therefore be omitted
only when the label also omits it (the label is the truth for which parameters
matter); declared required parameters must always be present or the format gate
fails.  A tool with an empty parameter schema has no argument information, so
its argument values are not type-checked (but calls to undeclared tools are).

An empty trajectory never scores.  All scores stay in the frozen ``[0, 1]``
range (CONTRACTS C4).
"""

from __future__ import annotations

from . import base

TASK_TYPE = "st_fc"
MAX_SCORE = 1.0


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``st_fc`` task/trajectory pair."""
    gt_calls = base.flatten_gt_calls(task)
    output_calls = base.all_calls(trajectory)
    match = base.match_calls(output_calls, gt_calls)
    # st_fc is single-turn *prediction*: tools are never executed (env.type ==
    # "none"), so emitted calls legitimately have no tool_results (C14 / D20).
    format_score, format_info = base.format_report(task, trajectory, expect_tool_results=False)
    gate_ok = base.format_ok(format_info)
    all_matched = bool(match["all_gt_matched"] and match["no_extra_calls"])
    score = MAX_SCORE if (all_matched and gate_ok) else 0.0

    info = {
        "all_calls_matched": all_matched,
        "all_gt_matched": match["all_gt_matched"],
        "no_extra_calls": match["no_extra_calls"],
        "format_ok": gate_ok,
        "format_score": format_score,
        "n_gt_calls": match["n_gt"],
        "n_output_calls": match["n_output"],
        "matched_calls": match["pairs"],
        "missed_calls": match["missed_calls"],
        "extra_calls": match["extra_calls"],
        "format_issues": format_info.get("issues", []),
        "empty_trajectory": base.is_empty_trajectory(trajectory),
        "gate": "all_gt_matched and no_extra_calls and format_ok",
    }
    return base.make_result(score, info, task_type=TASK_TYPE)
