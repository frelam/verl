"""``irrelevance`` reward - hammer-style "no tool needed" negatives (plan §5).

``score = 1.0`` when the policy emitted **no tool call at all**, else ``0.0``.
This is the literal plan rule ("未调用任何工具 = 1，否则 0"): a *decision*
reward about whether to invoke a tool, not a reply-quality reward, so a direct
answer and a refusal score identically here.

Boundary vs ``abstain`` (plan §1.5, §5): ``irrelevance`` (hammer) asks "should
a tool be called?" and accepts any no-call reply; ``abstain`` (When2Call)
additionally asks "does the reply correctly ask for missing information /
declare no suitable tool instead of fabricating an answer?" - see
``abstain.py``.  An abstain sample scored by ``irrelevance`` would get 1.0 for
a fabricated direct answer; the converse is not true (a correct question
scores 1.0 under both).

Reads only ``trajectory.turns[*].assistant.tool_calls`` (CONTRACTS C4).
"""

from __future__ import annotations

from . import base

TASK_TYPE = "irrelevance"
MAX_SCORE = 1.0


def reward(task: dict, trajectory: dict) -> dict:
    """Pure reward for one ``irrelevance`` task/trajectory pair."""
    calls = base.all_calls(trajectory)
    score = 0.0 if calls else MAX_SCORE
    info = {
        "n_output_calls": len(calls),
        "called_calls": [base.call_repr(call) for call in calls[:10]],
        "empty_trajectory": base.is_empty_trajectory(trajectory),
        "rule": "no tool call = 1, any tool call = 0 (plan §5)",
        "boundary": (
            "decision-only reward: direct answer and refusal score the same; "
            "see abstain for reply-quality scoring of 'information is missing -> ask'"
        ),
    }
    return base.make_result(score, info, task_type=TASK_TYPE)
