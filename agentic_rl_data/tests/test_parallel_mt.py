from __future__ import annotations

from agentic_rl_data.generators.parallel_mt_tasks import generate_tasks
from agentic_rl_data.schema import validate_task


def test_parallel_generator_covers_domains_and_is_deterministic() -> None:
    first = list(generate_tasks(3, seed=17))
    second = list(generate_tasks(3, seed=17))
    assert first == second
    assert {task["meta"]["domain"] for task in first} == {"filesystem", "retail", "trading"}
    for task in first:
        assert validate_task(task) == []
        assert task["source"] == "parallel_mt"
        assert task["meta"]["parallel_group_count"] == 1
        required = task["env"]["config"]["required_parallel"][0]
        assert 2 <= len(required) <= 4
        assert required in task["reference"]["gt_calls"]


def test_parallel_generator_varies_tool_menu_and_schema_style() -> None:
    tasks = list(generate_tasks(4, seed=23))
    assert len(tasks[3]["tools"]) == 20
    assert tasks[3]["meta"]["distractor_count"] > 0
    assert {task["meta"]["schema_style"] for task in tasks} != {"canonical"}
    for task in tasks:
        assert task["meta"]["canonical_tools"]
