from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import pytest

from agentic_rl_data import schema
from agentic_rl_data.envs.base import BaseEnv, make_tool_result
from agentic_rl_data.scripts import dedup, decontam, replay_check
from agentic_rl_data.scripts.report import compute_distributions, render_markdown


def _tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Public description for {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def _task(
    task_id: str,
    user: str = "Initial request",
    *,
    stages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    aliases: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    gt_calls: list[list[dict[str, Any]]] | None = None,
    gt_turns: list[dict[str, Any]] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env_config = dict(config or {"impl": "stateful", "domain": "filesystem"})
    if stages is not None:
        env_config["dialogue"] = {"stages": copy.deepcopy(stages)}
    if aliases is not None:
        env_config["tool_aliases"] = dict(aliases)
    task_meta = {
        "max_turns": 15,
        "difficulty": None,
        "pass_rate": None,
        "license": None,
        "gt_returns_source": "config",
    }
    task_meta.update(dict(meta or {}))
    return {
        "id": task_id,
        "source": "diversity-test",
        "task_type": "mt_fc",
        "messages": [{"role": "user", "content": user}],
        "tools": copy.deepcopy(tools or []),
        "env": {"type": "mock_fc", "config": env_config},
        "reference": {
            "final_answer": None,
            "final_answers": None,
            "gt_calls": copy.deepcopy(gt_calls if gt_calls is not None else [[]]),
            "gt_turns": copy.deepcopy(gt_turns),
            "tool_returns": None,
            "unit_tests": None,
            "final_state": None,
        },
        "meta": task_meta,
    }


def test_task_user_texts_preserves_legacy_key_and_adds_staged_users() -> None:
    legacy = _task("legacy", "Weather in Paris?", tools=[_tool("weather")])
    assert schema.task_user_texts(legacy) == ["Weather in Paris?"]
    legacy_payload = (
        f"{schema.normalize_query('Weather in Paris?')}\x1f"
        f"{schema.tools_signature(legacy['tools'])}"
    )
    assert schema.task_dedup_key(legacy) == hashlib.sha256(
        legacy_payload.encode("utf-8")
    ).hexdigest()

    dialogue = _task(
        "dialogue",
        "First instruction",
        stages=[{"goal": []}, {"next_user": "Second instruction"}, {"next_user": "Third instruction"}],
    )
    assert schema.task_user_texts(dialogue) == [
        "First instruction",
        "Second instruction",
        "Third instruction",
    ]
    assert [message["content"] for message in dialogue["messages"]] == ["First instruction"]


def test_exact_and_minhash_dedup_consume_staged_user_text() -> None:
    first = _task("first", "Shared opening", stages=[{"next_user": "Use Paris"}])
    different_future = _task("second", "Shared opening", stages=[{"next_user": "Use Tokyo"}])
    duplicate = copy.deepcopy(first)
    duplicate["id"] = "duplicate"

    kept, dropped = dedup.exact_dedup([first, different_future, duplicate])
    assert [task["id"] for task in kept] == ["first", "second"]
    assert [item["id"] for item in dropped] == ["duplicate"]

    assert dedup.task_text(first) == "shared opening\nuse paris"
    assert dedup.task_text(different_future) == "shared opening\nuse tokyo"
    assert dedup.task_shingles(first) != dedup.task_shingles(different_future)
    near_kept, near_dropped = dedup.near_dedup(
        [first, duplicate], threshold=0.9, num_perm=32, bands=8
    )
    assert [task["id"] for task in near_kept] == ["first"]
    assert [item["id"] for item in near_dropped] == ["duplicate"]


def test_decontam_checks_later_user_and_public_and_canonical_tool_names() -> None:
    follow_up = "this held out follow up instruction must never appear in any training example"
    index = decontam.build_index(
        [
            {
                "id": "eval-1",
                "question": follow_up,
                "function": [_tool("canonical_handler")],
            }
        ]
    )

    later_user = _task(
        "later-user",
        "A completely unrelated opening about gardening",
        stages=[{"next_user": follow_up}],
    )
    match = decontam.match_task(later_user, index)
    assert match["contaminated"]
    assert match["query_ngram_hits"] >= 1
    kept, dropped = decontam.decontaminate([later_user], index)
    assert kept == []
    assert [item["id"] for item in dropped] == ["later-user"]

    public = _task("public-name", "Clean request", tools=[_tool("canonical_handler")])
    assert decontam.match_task(public, index)["tool_name_hits"] == ["canonical_handler"]

    aliased = _task(
        "canonical-alias",
        "Clean request",
        tools=[_tool("publicSearch")],
        aliases={"publicSearch": "canonical_handler"},
    )
    alias_match = decontam.match_task(aliased, index)
    assert alias_match["contaminated"]
    assert alias_match["tool_name_hits"] == ["canonical_handler"]
    assert alias_match["matched_ids"] == ["eval-1"]


def test_report_counts_optional_diversity_metadata_only_when_present() -> None:
    tasks = [
        _task(
            "report-1",
            meta={
                "domain": "vehicle",
                "scenario": "booking",
                "schema_style": "camelCase",
                "user_turn_count": 3,
                "parallel_group_count": 2,
            },
        ),
        _task(
            "report-2",
            meta={
                "domain": "vehicle",
                "scenario": "cancellation",
                "schema_style": "snake_case",
                "user_turn_count": 1,
                "parallel_group_count": 0,
            },
        ),
        _task("report-3"),
    ]
    dist = compute_distributions(tasks)
    assert dist["domains"] == {"vehicle": 2}
    assert dist["scenarios"] == {"booking": 1, "cancellation": 1}
    assert dist["schema_styles"] == {"camelCase": 1, "snake_case": 1}
    assert dist["user_turn_counts"] == {"1": 1, "3": 1}
    assert dist["parallel_group_counts"] == {"0": 1, "2": 1}

    markdown = render_markdown(
        {
            "checks": {},
            "distributions": {"train": dist, "heldout": {}},
            "mixture": {},
            "steps": {},
            "not_run": [],
            "strict_failures": [],
        }
    )
    assert "meta.domain" in markdown
    assert '"vehicle": 2' in markdown
    assert "meta.parallel_group_count" in markdown


class _FakeDialogueEnv(BaseEnv):
    env_type = "mock_fc"
    impl = "stateful"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        self.actions: list[dict[str, Any]] = []
        self._next_users = list(config.get("test_next_users") or [])
        self._num_actions = len(task["reference"]["gt_calls"])

    def _handle_action(
        self, action: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any], str | None]:
        index = self._steps - 1
        self.actions.append(copy.deepcopy(action))
        results = [
            make_tool_result(call["name"], "ok", ok=True, call_id=call["id"])
            for call in action["tool_calls"]
        ]
        info: dict[str, Any] = {}
        if index < len(self._next_users) and self._next_users[index] is not None:
            info["next_user_message"] = self._next_users[index]
        done = index == self._num_actions - 1
        final_answer = action["content"] if done and not action["tool_calls"] else None
        return results, done, info, final_answer

    def _env_state_fields(self) -> dict[str, Any]:
        return {
            "goal_satisfied": self._done,
            "state": {},
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
        }


def _dialogue_replay_task() -> dict[str, Any]:
    return _task(
        "dialogue-replay",
        "Start",
        stages=[{"next_user": "Paris, please."}],
        tools=[_tool("lookup")],
        config={
            "impl": "stateful",
            "domain": "filesystem",
            "test_next_users": [None, "Paris, please.", None],
        },
        gt_calls=[
            [{"name": "lookup", "arguments": {"key": "x"}}],
            [],
            [],
        ],
        gt_turns=[
            {"content": "Looking up the value."},
            {"content": "Which city?", "next_user": "Paris, please."},
            {"content": "Done."},
        ],
    )


def test_dialogue_replay_uses_every_reference_action_and_validates_next_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeDialogueEnv] = []

    def fake_build_env(task: Mapping[str, Any], *, impl: str | None = None) -> _FakeDialogueEnv:
        env = _FakeDialogueEnv()
        created.append(env)
        return env

    monkeypatch.setattr(replay_check, "build_env", fake_build_env)
    task = _dialogue_replay_task()
    record = replay_check.check_task(task, reward_fn=None)
    assert record["ok"], record
    env = created[0]
    assert len(env.actions) == 3
    assert env.actions[0]["content"] == "Looking up the value."
    assert env.actions[1]["content"] == "Which city?"
    assert env.actions[1]["tool_calls"] == []
    assert env.actions[2]["content"] == "Done."
    assert env.trajectory()["final_answer"] == "Done."
    assert env.trajectory()["num_turns"] == 3

    mismatch = _dialogue_replay_task()
    mismatch["env"]["config"]["test_next_users"] = [None, "London instead", None]
    mismatch_record = replay_check.check_task(mismatch, reward_fn=None)
    assert mismatch_record["ok"] is False
    assert mismatch_record["stage"] == "next_user"
    assert mismatch_record["expected"] == "Paris, please."
    assert mismatch_record["actual"] == "London instead"


def test_dialogue_replay_requires_aligned_gt_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_check, "build_env", lambda task, *, impl=None: _FakeDialogueEnv())
    task = _dialogue_replay_task()
    task["reference"]["gt_turns"] = task["reference"]["gt_turns"][:2]
    record = replay_check.check_task(task, reward_fn=None)
    assert record["ok"] is False
    assert record["stage"] == "reference"
    assert "one entry per gt_calls action" in record["reason"]


def test_ordinary_replay_path_still_appends_reference_final_answer() -> None:
    task = _task(
        "ordinary-replay",
        tools=[_tool("lookup")],
        config={"impl": "replay"},
        gt_calls=[[{"name": "lookup", "arguments": {"key": "x"}}]],
        meta={"gt_returns_source": "reference"},
    )
    task["reference"]["tool_returns"] = [
        [{"name": "lookup", "content": "value", "ok": True}]
    ]
    task["reference"]["final_answer"] = "Finished."
    record = replay_check.check_task(task, impl="replay", reward_fn=None)
    assert record["ok"], record
    assert record["replay_ok"] is True
