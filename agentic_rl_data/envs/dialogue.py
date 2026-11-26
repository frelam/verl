"""Stage controller for scripted, interleaved-user stateful tasks.

The controller deliberately has no dependency on ``mock_fc``: goal evaluation is
injected by the caller.  It owns only dialogue semantics; the domain state and
all tool execution remain in ``MockFCEnv``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


CheckResults = list[dict[str, Any]]
GoalEvaluator = Callable[[Any, Mapping[str, Any]], CheckResults]


def _checks_pass(results: CheckResults) -> bool:
    return all(item.get("ok") is True for item in results)


def _slot_matches(content: str, slots: Any) -> bool:
    """Each slot may be a string or a list of accepted surface forms."""
    text = content.lower()
    for slot in slots or []:
        alternatives = slot if isinstance(slot, (list, tuple)) else [slot]
        alternatives = [str(item).strip().lower() for item in alternatives if str(item).strip()]
        if alternatives and not any(item in text for item in alternatives):
            return False
    return True


def _reply_matches(stage: Mapping[str, Any], content: str, reference_answer: str | None) -> tuple[bool, str]:
    expected = str(stage.get("reply") or "none")
    if expected in ("", "none"):
        return True, "ok"
    if not content.strip():
        return False, "empty reply"
    if expected == "answer_tag":
        from .mock_fc import answer_tag_matches

        answer = stage.get("expected_answer", stage.get("answer", reference_answer))
        return answer_tag_matches(content, answer), f"expected answer_tag={answer!r}"
    if expected not in ("request_info", "no_valid_tools"):
        return False, f"unknown reply gate {expected!r}"

    from ..rewards.abstain import classify_reply

    actual = classify_reply(content)
    if actual != expected:
        return False, f"reply class {actual!r}, expected {expected!r}"
    if expected == "request_info" and not _slot_matches(content, stage.get("slot_keywords")):
        return False, "clarification does not ask for every required slot"
    return True, "ok"


class DialogueController:
    """Track stage goals, evidence, reply gates and latched violations."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        initial_goal: list[Mapping[str, Any]],
        initial_state: Mapping[str, Any],
        evaluate: GoalEvaluator,
        reference_answer: str | None,
    ) -> None:
        raw_stages = config.get("stages") or []
        if not isinstance(raw_stages, list) or not all(isinstance(item, Mapping) for item in raw_stages):
            raise ValueError("env.config.dialogue.stages must be a list of mappings")
        self.stages: list[Mapping[str, Any]] = list(raw_stages)
        self.global_invariants = config.get("invariants") or []
        self.evaluate = evaluate
        self.reference_answer = reference_answer
        self.index = 0
        self.complete = not self.stages
        self.violations: list[str] = []
        self.results: list[dict[str, Any]] = []
        self.successful_tools: set[str] = set()
        self._initial_goal = initial_goal
        self._baseline: CheckResults = []
        self._begin_stage(initial_state)

    @property
    def stage(self) -> Mapping[str, Any]:
        if not self.stages:
            return {}
        return self.stages[min(self.index, len(self.stages) - 1)]

    @property
    def goal(self) -> list[Mapping[str, Any]]:
        if not self.stages:
            return self._initial_goal
        goal = self.stage.get("goal")
        return list(goal) if isinstance(goal, list) else []

    def _begin_stage(self, state: Mapping[str, Any]) -> None:
        self.successful_tools = set()
        self._baseline = self.evaluate(self.stage.get("must_change") or [], state)

    def _invariant_results(self, state: Mapping[str, Any]) -> CheckResults:
        checks = list(self.global_invariants) + list(self.stage.get("invariants") or [])
        return self.evaluate(checks, state)

    def on_tool_calls(self, call_info: list[dict[str, Any]], state: Mapping[str, Any]) -> None:
        if self.complete:
            return
        kind = str(self.stage.get("kind") or "execute")
        allow_tools = bool(self.stage.get("allow_tools", kind not in ("clarify", "abstain")))
        if not allow_tools:
            self.violations.append(f"stage {self.index}: tool call before clarification/refusal")
        for item in call_info:
            if item.get("ok") is True and isinstance(item.get("name"), str):
                self.successful_tools.add(item["name"])
        failed = [item["name"] for item in self._invariant_results(state) if not item.get("ok")]
        if failed:
            self.violations.append(f"stage {self.index}: invariants failed {failed[:5]}")

    def advance(self, content: str, state: Mapping[str, Any]) -> tuple[bool, str | None, str]:
        """Consume a no-call reply; return ``(advanced, next_user, detail)``."""
        if self.complete:
            return False, None, "dialogue already complete"
        if self.violations:
            reason = self.violations[-1]
            self.results.append({"stage": self.index, "ok": False, "detail": reason})
            return False, None, reason

        invariant_results = self._invariant_results(state)
        goal_results = self.evaluate(self.stage.get("goal") or [], state)
        change_results = self.evaluate(self.stage.get("must_change") or [], state)
        required = {str(item) for item in self.stage.get("required_tools") or []}
        missing_tools = sorted(required - self.successful_tools)
        reply_ok, reply_detail = _reply_matches(self.stage, content, self.reference_answer)

        ok = True
        detail = "ok"
        if not _checks_pass(invariant_results):
            ok, detail = False, "stage invariants not satisfied"
        elif not _checks_pass(goal_results):
            ok, detail = False, "stage goal not satisfied"
        elif self._baseline and _checks_pass(self._baseline):
            ok, detail = False, "must_change goal was already true at stage entry"
        elif not _checks_pass(change_results):
            ok, detail = False, "required state change did not happen"
        elif missing_tools:
            ok, detail = False, f"required successful tools missing: {missing_tools}"
        elif not reply_ok:
            ok, detail = False, reply_detail

        self.results.append({"stage": self.index, "ok": ok, "detail": detail})
        if not ok:
            self.violations.append(f"stage {self.index}: {detail}")
            return False, None, detail

        completed = self.stage
        if self.index + 1 >= len(self.stages):
            self.complete = True
            return True, None, "dialogue complete"
        self.index += 1
        self._begin_stage(state)
        next_user = completed.get("next_user", completed.get("user"))
        if not isinstance(next_user, str) or not next_user.strip():
            self.violations.append(f"stage {self.index - 1}: missing next_user")
            return False, None, "completed stage has no next_user"
        return True, next_user, "stage complete"

    def terminal_checks(self, state: Mapping[str, Any]) -> CheckResults:
        if not self.stages:
            return []
        checks = self.evaluate(self.global_invariants, state)
        checks.append(
            {
                "name": "dialogue_complete",
                "ok": self.complete and not self.violations,
                "detail": f"cursor={self.index}/{len(self.stages)} violations={self.violations[-3:]}",
            }
        )
        return checks

    def snapshot(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "cursor": self.index,
            "stage_count": len(self.stages),
            "violations": list(self.violations),
            "results": [dict(item) for item in self.results],
            "successful_tools": sorted(self.successful_tools),
        }
