"""LLM Judge reward — re-exports from uni_agent.reward.llm_judge.

All scoring logic lives in ``uni_agent/reward/llm_judge.py`` following
the verl pluggable ``custom_reward_function`` pattern.  This module is
kept as a thin redirect for backward compatibility.
"""

from uni_agent.reward.llm_judge import (  # noqa: F401
    compute_score,
    judge_single,
    load_judge_prompt,
)
