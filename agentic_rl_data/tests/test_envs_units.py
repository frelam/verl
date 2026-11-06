"""Unit tests for search / python sandbox / code_tests / replay_check.

**Not executed in the authoring session** (no working shell on this machine:
``pwsh`` fails with ``0xC0000142``).  Run on the user machine with::

    cd D:\\lhy\\verl\\verl-toolRL
    python -m pytest agentic_rl_data/tests/test_envs_units.py -q

Optional-dependency tests (FastAPI service) skip themselves when
``fastapi`` / ``httpx`` are absent.  The rest is stdlib + pytest.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

import pytest

from agentic_rl_data import schema
from agentic_rl_data.envs.code_tests import (
    CodeTestsEnv,
    _collect_failures,
    extract_code,
    parse_test_output,
    run_unit_tests,
)
from agentic_rl_data.envs.python_env import (
    PythonEnv,
    build_sandbox_config,
    run_python,
)
from agentic_rl_data.envs.search import (
    InMemorySearchBackend,
    SearchBackendError,
    SearchClient,
    SearchEnv,
    create_app,
    load_documents,
    normalise_search_config,
)
from agentic_rl_data.scripts.replay_check import check_task, main as replay_main


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _reference(**overrides: Any) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "final_answer": None,
        "final_answers": None,
        "gt_calls": [],
        "gt_turns": None,
        "tool_returns": None,
        "unit_tests": None,
        "final_state": None,
    }
    reference.update(overrides)
    return reference


def make_task(
    *,
    task_type: str = "mt_fc",
    env_type: str = "mock_fc",
    config: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_turns: int = 15,
    task_id: str = "unit-1",
    gt_returns_source: str | None = None,
) -> dict[str, Any]:
    if gt_returns_source is None and task_type == "mt_fc":
        # schema v2.2: required for mt_fc, enum singular "reference".
        gt_returns_source = "reference"
    meta: dict[str, Any] = {
        "max_turns": max_turns,
        "difficulty": None,
        "pass_rate": None,
        "license": None,
    }
    if gt_returns_source is not None:
        meta["gt_returns_source"] = gt_returns_source
    task = {
        "id": task_id,
        "source": "unit-test",
        "task_type": task_type,
        "messages": [{"role": "user", "content": "do the task"}],
        "tools": list(tools or []),
        "env": {"type": env_type, "config": dict(config or {})},
        "reference": _reference(**(reference or {})),
        "meta": meta,
    }
    errors = schema.validate_task(task)
    assert not errors, f"fixture task is schema-invalid: {errors}"
    return task


def tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"tool {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": dict(arguments)}


CORPUS = [
    {
        "doc_id": "doc-fr",
        "title": "France",
        "text": "France is a country in Western Europe. Its capital is Paris. The Eiffel Tower is in Paris.",
    },
    {
        "doc_id": "doc-de",
        "title": "Germany",
        "text": "Germany is a country in Central Europe. Its capital is Berlin.",
    },
    {
        "doc_id": "doc-wiki",
        "title": "Wikipedia",
        "text": "Wikipedia is a free online encyclopedia edited by volunteers.",
    },
]


# ---------------------------------------------------------------------------
# InMemorySearchBackend
# ---------------------------------------------------------------------------


def test_inmemory_backend_is_deterministic_and_relevant() -> None:
    backend = InMemorySearchBackend(CORPUS, snippet_chars=200)
    first = backend.search("capital of France", 2)
    second = backend.search("capital of France", 2)
    assert first == second
    assert len(first) == 2
    assert first[0]["doc_id"] == "doc-fr"
    assert first[0]["title"] == "France"
    assert "Paris" in first[0]["snippet"]
    assert isinstance(first[0]["score"], float)
    assert backend.doc_count == 3


def test_inmemory_backend_pagination_and_empty_queries() -> None:
    backend = InMemorySearchBackend(CORPUS, snippet_chars=80)
    assert backend.search("zzzz-unknown-token", 3) == []
    assert backend.search("   ", 3) == []
    assert backend.search("capital", 1) == backend.search("capital", 1)[:1]
    top_one = backend.search("country", 1)
    top_two = backend.search("country", 2)
    assert len(top_one) == 1 and len(top_two) == 2
    assert top_two[0] == top_one[0]


def test_inmemory_backend_fetch() -> None:
    backend = InMemorySearchBackend(CORPUS)
    document = backend.fetch("doc-de")
    assert document["title"] == "Germany"
    assert "Berlin" in document["text"]
    with pytest.raises(SearchBackendError):
        backend.fetch("missing-doc")


def test_load_documents_from_jsonl(tmp_path: Any) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in CORPUS) + "\n",
        encoding="utf-8",
    )
    docs = load_documents(path)
    assert len(docs) == 3
    assert docs[0]["doc_id"] == "doc-fr"


# ---------------------------------------------------------------------------
# SearchEnv
# ---------------------------------------------------------------------------


def _search_task(config: Mapping[str, Any] | None = None, max_turns: int = 15) -> dict[str, Any]:
    return make_task(
        task_type="search_qa",
        env_type="search",
        config={"backend": "inmemory", "docs": CORPUS, **(dict(config or {}))},
        tools=[tool("search"), tool("fetch")],
        max_turns=max_turns,
    )


def test_search_env_search_and_fetch() -> None:
    env = SearchEnv(_search_task())
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [call("search", query="capital of France", top_k=2)]}
    )
    assert done is False
    payload = json.loads(obs["tool_results"][0]["content"])
    assert len(payload["results"]) == 2
    assert payload["results"][0]["doc_id"] == "doc-fr"
    assert payload["results"][0]["snippet"]

    obs, done, _ = env.step({"content": "", "tool_calls": [call("fetch", doc_id="doc-fr")]})
    document = json.loads(obs["tool_results"][0]["content"])
    assert document["doc_id"] == "doc-fr"
    assert "Paris" in document["text"]
    assert env.env_state["extras"]["fetched_docs"] == ["doc-fr"]

    _, done, info = env.step({"content": "Paris"})
    assert done is True
    assert info["final_answer"] == "Paris"


def test_search_env_unknown_tool_is_recoverable() -> None:
    env = SearchEnv(_search_task())
    env.reset()
    obs, done, _ = env.step({"content": "", "tool_calls": [call("get_weather", city="Paris")]})
    assert done is False
    assert obs["tool_results"][0]["ok"] is False
    assert "unknown search tool" in obs["tool_results"][0]["content"]


def test_search_env_caps_turns_at_eight_rounds() -> None:
    env = SearchEnv(_search_task(max_turns=15))
    env.reset()
    assert env.max_turns == 8
    for index in range(8):
        obs, done, info = env.step(
            {"content": "", "tool_calls": [call("search", query=f"query {index}")]}
        )
        if index < 7:
            assert done is False, f"episode ended early at round {index}"
        else:
            assert done is True
            assert info["timeout"] is True
    state = env.env_state
    assert len(state["extras"]["queries"]) == 8
    assert env.trajectory()["done"] is True


def test_search_env_curriculum_override_can_lower_rounds() -> None:
    env = SearchEnv(_search_task())
    env.reset(env.task, max_turns=5)
    assert env.max_turns == 5
    for index in range(5):
        _, done, info = env.step(
            {"content": "", "tool_calls": [call("search", query=f"q{index}")]}
        )
    assert done is True
    assert info["timeout"] is True


class _FailingBackend:
    name = "failing"

    @property
    def doc_count(self) -> int:
        return 0

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        raise SearchBackendError("simulated outage", retryable=True)

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        raise SearchBackendError("simulated outage", retryable=True)

    def close(self) -> None:
        return None


def test_search_env_backend_failure_is_a_failed_tool_result() -> None:
    backend = _FailingBackend()
    env = SearchEnv(
        _search_task({"backend": backend, "own_backend": False})
    )
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [call("search", query="capital")]}
    )
    assert done is False
    assert obs["tool_results"][0]["ok"] is False
    assert "backend unavailable" in obs["tool_results"][0]["content"]
    assert info.get("backend_error") is True
    assert env.env_state["extras"]["backend_errors"] == 1
    _, done, _ = env.step({"content": "no results"})
    assert done is True


def test_search_config_aliases_from_loader_form() -> None:
    # D12 / CONTRACTS C11: transitional loader form must keep working.
    canonical = normalise_search_config(
        {
            "backend": "local_wikipedia_e5",
            "retriever": {"url": "http://127.0.0.1:8899", "topk": 2},
        }
    )
    # A URL with no local index files means the E5 index is behind HTTP.
    assert canonical["backend"] == "http"
    assert canonical["url"] == "http://127.0.0.1:8899"
    assert canonical["top_k"] == 2

    # Without a URL the alias stays an in-process E5 backend.
    assert normalise_search_config({"backend": "local_wikipedia_e5"})["backend"] == "e5"
    assert (
        normalise_search_config(
            {
                "backend": "local_wikipedia_e5",
                "docs_path": "docs.jsonl",
                "index_path": "index.npz",
            }
        )["backend"]
        == "e5"
    )

    env = SearchEnv(
        make_task(
            task_type="search_qa",
            env_type="search",
            config={
                "backend": "local_wikipedia_e5",
                "retriever": {"url": "http://127.0.0.1:8899", "topk": 2},
                "task_deadline_s": 10,
            },
            max_turns=15,
        )
    )
    env.reset()
    assert env.max_turns == 8
    assert env.env_state["extras"]["top_k"] == 2
    assert env.env_state["extras"]["retriever_aliased"] is True
    assert env.env_state["extras"]["backend"] == "http"

    # retriever can also carry the in-memory backend itself.
    env = SearchEnv(
        make_task(
            task_type="search_qa",
            env_type="search",
            config={"retriever": {"backend": "inmemory", "docs": CORPUS, "topk": 2}},
        )
    )
    env.reset()
    obs, _, _ = env.step({"content": "", "tool_calls": [call("search", query="capital of France")]})
    payload = json.loads(obs["tool_results"][0]["content"])
    assert payload["results"][0]["doc_id"] == "doc-fr"


def test_search_env_does_not_raise_when_http_service_is_down() -> None:
    # A configured-but-unreachable service must not fail env construction: the
    # tool call degrades to ok=False (rollout keeps running).
    env = SearchEnv(
        make_task(
            task_type="search_qa",
            env_type="search",
            config={
                "backend": "http",
                "url": "http://127.0.0.1:9",
                "request_timeout_s": 0.2,
                "max_retries": 0,
            },
        )
    )
    env.reset()
    obs, done, info = env.step({"content": "", "tool_calls": [call("search", query="q")]})
    assert done is False
    assert obs["tool_results"][0]["ok"] is False
    assert info.get("backend_error") is True


def test_search_env_requires_backend_configuration() -> None:
    from agentic_rl_data.envs.base import EnvError

    with pytest.raises(EnvError):
        SearchEnv(make_task(task_type="search_qa", env_type="search", config={}))


class _SlowSearchBackend:
    name = "slow-search"

    @property
    def doc_count(self) -> int:
        return 1

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        time.sleep(0.35)
        return [{"doc_id": "d1", "title": "t", "snippet": "s", "score": 1.0}]

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        time.sleep(0.35)
        return {"doc_id": doc_id, "title": "t", "text": "body"}

    def close(self) -> None:
        return None


def test_search_env_task_deadline_returns_backend_error_not_stall() -> None:
    backend = _SlowSearchBackend()
    env = SearchEnv(
        _search_task({"backend": backend, "own_backend": False, "task_deadline_s": 0.2})
    )
    env.reset()
    obs, _, _ = env.step({"content": "", "tool_calls": [call("search", query="first")]})
    assert obs["tool_results"][0]["ok"] is True  # first call starts before the deadline
    obs, done, info = env.step({"content": "", "tool_calls": [call("search", query="second")]})
    assert done is False  # recoverable failure, episode continues
    assert obs["tool_results"][0]["ok"] is False
    assert "deadline" in obs["tool_results"][0]["content"]
    assert info["deadline_exceeded"] is True
    assert info["backend_error"] is True
    extras = env.env_state["extras"]
    assert extras["deadline_exceeded"] is True
    assert extras["task_deadline_s"] == 0.2
    _, done, _ = env.step({"content": "answer"})
    assert done is True


# ---------------------------------------------------------------------------
# SearchClient retries / timeouts
# ---------------------------------------------------------------------------


def test_search_client_retries_retryable_errors_then_succeeds() -> None:
    attempts: list[int] = []

    def transport(method: str, url: str, payload: Any, timeout: float) -> Any:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 2:
            raise SearchBackendError("temporary", retryable=True)
        return {"results": [{"doc_id": "d1", "title": "t", "snippet": "s", "score": 1.0}]}

    client = SearchClient(
        "http://127.0.0.1:9",
        max_retries=2,
        backoff_s=0.0,
        max_backoff_s=0.0,
        transport=transport,
    )
    results = client.search("q", 1)
    assert len(attempts) == 3
    assert results[0]["doc_id"] == "d1"


def test_search_client_does_not_retry_non_retryable_errors() -> None:
    attempts: list[int] = []

    def transport(method: str, url: str, payload: Any, timeout: float) -> Any:
        attempts.append(1)
        raise SearchBackendError("bad request", retryable=False)

    client = SearchClient(
        "http://127.0.0.1:9", max_retries=3, backoff_s=0.0, transport=transport
    )
    with pytest.raises(SearchBackendError):
        client.search("q", 1)
    assert len(attempts) == 1


def test_search_client_exhausts_retries() -> None:
    attempts: list[int] = []

    def transport(method: str, url: str, payload: Any, timeout: float) -> Any:
        attempts.append(1)
        raise SearchBackendError("unavailable", retryable=True)

    client = SearchClient(
        "http://127.0.0.1:9", max_retries=2, backoff_s=0.0, transport=transport
    )
    with pytest.raises(SearchBackendError) as excinfo:
        client.search("q", 1)
    assert "retries exhausted" in str(excinfo.value)
    assert len(attempts) == 3


def test_search_client_fetch_and_malformed_response() -> None:
    def transport(method: str, url: str, payload: Any, timeout: float) -> Any:
        if "/fetch" in url:
            return {"doc_id": "d1", "title": "T", "text": "body"}
        return {"unexpected": True}

    client = SearchClient("http://127.0.0.1:9", transport=transport)
    assert client.fetch("d1")["text"] == "body"
    with pytest.raises(SearchBackendError):
        client.search("q", 1)


# ---------------------------------------------------------------------------
# HTTP service (optional fastapi)
# ---------------------------------------------------------------------------


def test_http_service_endpoints() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient  # type: ignore

    backend = InMemorySearchBackend(CORPUS, snippet_chars=120)
    application = create_app(backend, request_timeout_s=5.0, max_workers=4)
    with TestClient(application) as client:
        response = client.post("/search", json={"query": "capital of France", "top_k": 2})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["results"]) == 2
        assert payload["results"][0]["doc_id"] == "doc-fr"

        response = client.get("/fetch", params={"doc_id": "doc-fr"})
        assert response.status_code == 200
        assert "Paris" in response.json()["text"]

        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class _SlowBackend:
    name = "slow"

    @property
    def doc_count(self) -> int:
        return 0

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        time.sleep(0.5)
        return []

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        time.sleep(0.5)
        return {"doc_id": doc_id, "title": "", "text": ""}

    def close(self) -> None:
        return None


def test_http_service_enforces_server_side_timeout() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient  # type: ignore

    application = create_app(_SlowBackend(), request_timeout_s=0.05, max_workers=2)
    with TestClient(application) as client:
        response = client.post("/search", json={"query": "q", "top_k": 1})
        assert response.status_code == 503
        assert "timed out" in response.json()["detail"]


# ---------------------------------------------------------------------------
# python_env
# ---------------------------------------------------------------------------


def test_run_python_hello() -> None:
    result = run_python("print('hello from sandbox')")
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert "hello from sandbox" in result["stdout"]
    assert result["network_isolated"] is False
    assert "script" in result


def test_run_python_syntax_error() -> None:
    result = run_python("def broken(:\n    pass")
    assert result["exit_code"] != 0
    assert "SyntaxError" in result["stderr"] or "SyntaxError" in result["stdout"]


def test_run_python_timeout() -> None:
    result = run_python("import time\ntime.sleep(30)", timeout_s=0.5)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert result["duration_s"] < 10.0


def test_run_python_truncates_output() -> None:
    config = build_sandbox_config({"max_output_chars": 200})
    result = run_python("print('x' * 5000)", config=config)
    assert result["truncated"] is True
    assert len(result["stdout"]) <= 200


def test_run_python_rejects_empty_code() -> None:
    from agentic_rl_data.envs.base import EnvError

    with pytest.raises(EnvError):
        run_python("   ")


def test_python_env_protocol() -> None:
    task = make_task(task_type="math_ci", env_type="python", config={"timeout_s": 5})
    env = PythonEnv(task)
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [call("python", code="print(2 + 2)")]}
    )
    assert done is False
    payload = json.loads(obs["tool_results"][0]["content"])
    assert payload["exit_code"] == 0
    assert "4" in payload["stdout"]
    assert obs["tool_results"][0]["ok"] is True
    assert len(info["executions"]) == 1
    assert env.env_state["impl"] == "local"
    assert env.env_state["extras"]["sandbox"]["network_isolated"] is False

    obs, done, _ = env.step({"content": "", "tool_calls": [call("python", code="raise SystemExit(3)")]})
    assert obs["tool_results"][0]["ok"] is False
    payload = json.loads(obs["tool_results"][0]["content"])
    assert payload["exit_code"] == 3

    obs, done, _ = env.step({"content": "", "tool_calls": [call("search", query="x")]})
    assert obs["tool_results"][0]["ok"] is False

    _, done, info = env.step({"content": "4"})
    assert done is True
    assert info["timeout"] is False
    errors = schema.validate_trajectory(env.trajectory())
    assert not errors


# ---------------------------------------------------------------------------
# code_tests
# ---------------------------------------------------------------------------

UNITTEST_TESTS = """\
import unittest
from solution import add

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)
"""

GOOD_CODE = "def add(a, b):\n    return a + b\n"
BAD_CODE = "def add(a, b):\n    return a - b\n"


def test_parse_test_output_pytest_and_unittest() -> None:
    pytest_output = "2 passed in 0.02s"
    counts = parse_test_output(pytest_output, runner="pytest")
    assert counts == {"total": 2, "passed": 2, "failed": 0}

    counts = parse_test_output("1 failed, 1 passed in 0.12s", runner="pytest")
    assert counts["passed"] == 1
    assert counts["failed"] == 1
    assert counts["total"] == 2

    counts = parse_test_output("1 error, 1 passed", runner="pytest")
    assert counts["failed"] == 1

    unittest_output = "Ran 3 tests in 0.001s\n\nFAILED (failures=1, errors=1)"
    counts = parse_test_output(unittest_output, runner="unittest")
    assert counts["total"] == 3
    assert counts["failed"] == 2
    assert counts["passed"] == 1

    unittest_ok = "Ran 2 tests in 0.000s\n\nOK"
    counts = parse_test_output(unittest_ok, runner="unittest")
    assert counts == {"total": 2, "passed": 2, "failed": 0}


def test_extract_code_from_fenced_block() -> None:
    text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert extract_code(text).startswith("def add")
    assert extract_code("plain text") == "plain text"
    assert extract_code(None) == ""


def _code_task(unit_tests: str | None = UNITTEST_TESTS, **config: Any) -> dict[str, Any]:
    return make_task(
        task_type="code_exec",
        env_type="code_tests",
        config=dict(config),
        reference={"unit_tests": unit_tests},
        tools=[tool("run_tests")],
        max_turns=15,
        task_id="code-1",
    )


def test_code_tests_passing_solution_exposes_authoritative_env_state() -> None:
    env = CodeTestsEnv(_code_task())
    env.reset()
    obs, done, info = env.step(
        {"content": "", "tool_calls": [call("run_tests", code=GOOD_CODE)]}
    )
    assert done is False
    assert obs["tool_results"][0]["ok"] is True
    payload = json.loads(obs["tool_results"][0]["content"])
    assert payload["all_passed"] is True
    assert payload["passed"] == payload["total"]

    _, done, info = env.step({"content": "done"})
    assert done is True
    code_tests = env.env_state["code_tests"]
    assert set(code_tests) >= {
        "total", "passed", "failed", "all_passed", "infra_error",
        "exit_code", "duration_s", "truncated", "failures",
    }
    assert code_tests["all_passed"] is True
    assert code_tests["infra_error"] is False
    assert info["tests"]["all_passed"] is True  # framework log duplicate (D6)


def test_code_tests_failing_solution_never_leaks_code() -> None:
    env = CodeTestsEnv(_code_task())
    env.reset()
    obs, done, _ = env.step(
        {"content": "", "tool_calls": [call("run_tests", code=BAD_CODE)]}
    )
    assert done is False
    assert obs["tool_results"][0]["ok"] is False
    content = obs["tool_results"][0]["content"]
    assert "def add" not in content
    assert "return a - b" not in content
    payload = json.loads(content)
    assert payload["all_passed"] is False
    assert payload["failed"] >= 1
    assert payload["failures"]
    assert all(len(item["message"]) <= 200 for item in payload["failures"])

    _, done, _ = env.step({"content": "done"})
    code_tests = env.env_state["code_tests"]
    assert code_tests["all_passed"] is False
    assert code_tests["passed"] == code_tests["total"] - code_tests["failed"]


def test_code_tests_final_answer_with_fenced_code() -> None:
    env = CodeTestsEnv(_code_task())
    env.reset()
    answer = "Solution:\n```python\n" + GOOD_CODE + "```\n"
    _, done, info = env.step({"content": answer, "tool_calls": []})
    assert done is True
    assert env.env_state["code_tests"]["all_passed"] is True
    assert info["tests"]["all_passed"] is True


def test_code_tests_missing_unit_tests_is_reported_as_infra_failure() -> None:
    env = CodeTestsEnv(_code_task(unit_tests=None))
    env.reset()
    _, done, info = env.step({"content": "```python\n" + GOOD_CODE + "```", "tool_calls": []})
    assert done is True
    code_tests = env.env_state["code_tests"]
    assert code_tests["all_passed"] is False
    assert code_tests["infra_error"] is True  # T4 P1: harness failure, not wrong answer
    assert env.env_state["extras"]["tests_configured"] is False
    assert env.env_state["extras"]["infra_error"] is True


def test_code_tests_no_code_from_policy_is_not_an_infra_failure() -> None:
    env = CodeTestsEnv(_code_task())
    env.reset()
    _, done, info = env.step({"content": "", "tool_calls": []})
    assert done is True
    assert env.env_state["code_tests"]["all_passed"] is False
    assert env.env_state["code_tests"]["infra_error"] is False


def test_code_tests_zero_collected_tests_is_an_infra_failure() -> None:
    # A test file with no tests at all: the harness collected nothing.
    env = CodeTestsEnv(_code_task(unit_tests="x = 1\n"))
    env.reset()
    _, _, _ = env.step({"content": "", "tool_calls": [call("run_tests", code=GOOD_CODE)]})
    code_tests = env.env_state["code_tests"]
    assert code_tests["all_passed"] is False
    assert code_tests["infra_error"] is True
    assert code_tests["total"] == 0


def test_code_tests_timeout_is_a_failure() -> None:
    slow_tests = (
        "import unittest, time\n"
        "class TestSlow(unittest.TestCase):\n"
        "    def test_slow(self):\n"
        "        time.sleep(5)\n"
    )
    env = CodeTestsEnv(_code_task(unit_tests=slow_tests, timeout_s=0.5))
    env.reset()
    obs, _, _ = env.step(
        {"content": "", "tool_calls": [call("run_tests", code="x = 1\n")]}
    )
    assert obs["tool_results"][0]["ok"] is False
    tests = env.env_state["code_tests"]
    assert tests["all_passed"] is False
    assert tests["infra_error"] is False  # a hanging test is a policy failure
    assert tests["exit_code"] == -1


def test_code_tests_unknown_tool_and_empty_code() -> None:
    env = CodeTestsEnv(_code_task())
    env.reset()
    obs, _, _ = env.step({"content": "", "tool_calls": [call("search", query="x")]})
    assert obs["tool_results"][0]["ok"] is False
    assert "unknown code_exec tool" in obs["tool_results"][0]["content"]

    obs, _, _ = env.step({"content": "", "tool_calls": [call("run_tests")]})
    assert obs["tool_results"][0]["ok"] is False
    assert "code argument is required" in obs["tool_results"][0]["content"]


def test_code_tests_redacts_failure_messages() -> None:
    output = "FAILED test_solution.py::test_x - SECRET expected 42 but got 7"
    failures = _collect_failures(output, runner="pytest", redact_patterns=["SECRET"])
    assert failures
    assert "SECRET" not in failures[0]["message"]
    assert "[redacted]" in failures[0]["message"]


def test_run_unit_tests_helper() -> None:
    result = run_unit_tests(GOOD_CODE, UNITTEST_TESTS, runner="auto", timeout_s=10)
    assert result["all_passed"] is True
    assert result["passed"] == result["total"] == 2

    result = run_unit_tests(BAD_CODE, UNITTEST_TESTS, runner="auto", timeout_s=10)
    assert result["all_passed"] is False
    assert result["failed"] >= 1


# ---------------------------------------------------------------------------
# replay_check
# ---------------------------------------------------------------------------


def _replay_fixture(
    *,
    task_id: str = "rc-1",
    with_returns: bool = True,
    wrong_call: bool = False,
    mock_dir: Any | None = None,
) -> dict[str, Any]:
    gt_calls = [[{"name": "get_weather", "arguments": {"city": "Beijing"}}]]
    returns = [[{"name": "get_weather", "content": "sunny", "ok": True}]] if with_returns else None
    task = make_task(
        task_id=task_id,
        config={"impl": "replay"},
        reference={
            "gt_calls": gt_calls,
            "tool_returns": returns,
            "final_answer": "It is sunny.",
        },
        tools=[tool("get_weather")],
    )
    if wrong_call:
        task["reference"]["gt_calls"] = [[{"name": "get_weather", "arguments": {"city": "Paris"}}]]
    return task


def test_replay_check_passes_full_match_without_reward() -> None:
    record = check_task(_replay_fixture(), impl="replay", reward_fn=None)
    assert record["ok"] is True
    assert record["stage"] == "passed"
    assert record["replay_ok"] is True
    assert record["gt_returns_source"] == "reference"
    assert record["reward_used"] is False


def test_replay_check_fails_on_stateful_goal() -> None:
    task = make_task(
        task_id="rc-goal-fail",
        config={
            "impl": "stateful",
            "domain": "filesystem",
            "goal": [{"check": "file_exists", "path": "/missing.txt"}],
        },
        tools=[tool("touch")],
        reference={
            "gt_calls": [[{"name": "touch", "arguments": {"path": "/a.txt"}}]],
            "final_answer": "done",
        },
    )
    record = check_task(task, impl="stateful", reward_fn=None)
    assert record["ok"] is False
    assert record["stage"] == "goal"
    assert "goal_satisfied is False" in record["reason"]


def test_replay_check_reports_schema_invalid_task() -> None:
    task = make_task(config={"impl": "replay"})
    del task["reference"]["final_state"]
    record = check_task(task, impl="replay", reward_fn=None)
    assert record["ok"] is False
    assert record["stage"] == "schema"
    assert any("final_state" in error for error in record["details"])


def test_replay_check_reports_missing_meta_source_declaration() -> None:
    task = make_task(config={"impl": "replay"})
    del task["meta"]["gt_returns_source"]
    record = check_task(task, impl="replay", reward_fn=None)
    assert record["ok"] is False
    assert record["stage"] == "schema"
    assert any("gt_returns_source" in error for error in record["details"])


def test_replay_check_reports_gt_returns_missing_as_degraded() -> None:
    task = _replay_fixture(task_id="rc-missing")
    task["reference"]["tool_returns"] = [
        [{"name": "get_weather", "content": "sunny", "ok": True}],
        [{"name": "get_weather", "content": "extra", "ok": True}],
    ]
    task["reference"]["gt_calls"] = [
        [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        [{"name": "get_weather", "arguments": {"city": "Paris"}}],
    ]
    # Only one return group exists for two turns -> the second turn is missing.
    task["reference"]["tool_returns"] = task["reference"]["tool_returns"][:1]
    task["env"]["config"]["max_missing_ratio"] = 0.0
    record = check_task(task, impl="replay", reward_fn=None)
    assert record["ok"] is True  # replay still matches; data quality is degraded
    assert record["gt_returns_missing"] == 1
    assert record["degraded_replay"] is True
    assert record["gt_returns_source"] == "placeholder"
    assert any("no ground-truth return" in warning for warning in record["warnings"])


def test_replay_check_warns_on_meta_source_mismatch() -> None:
    task = _replay_fixture(task_id="rc-mismatch")
    task["meta"]["gt_returns_source"] = "gt_turns"  # runtime will use "reference"
    record = check_task(task, impl="replay", reward_fn=None)
    assert record["ok"] is True
    assert record["gt_returns_source_mismatch"] is True
    assert record["gt_returns_source"] == "reference"
    assert any("runtime wins" in warning for warning in record["warnings"])


def test_replay_check_requires_full_reward_score() -> None:
    def full_reward(task: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
        return {"score": 1.0, "info": {"verbal": "full"}}

    record = check_task(_replay_fixture(), impl="replay", reward_fn=full_reward)
    assert record["ok"] is True
    assert record["reward_used"] is True
    assert record["score"] == 1.0

    def partial_reward(task: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
        return {"score": 0.8, "info": {}}

    record = check_task(_replay_fixture(), impl="replay", reward_fn=partial_reward)
    assert record["ok"] is False
    assert record["stage"] == "reward_score"
    assert "0.8000" in record["reason"]


def test_replay_check_stateful_final_state_fallback() -> None:
    task = make_task(
        task_id="rc-stateful",
        config={"impl": "stateful", "domain": "filesystem"},
        tools=[tool("touch")],
        reference={
            "gt_calls": [[{"name": "touch", "arguments": {"path": "/a.txt"}}]],
            "final_state": {"files": {"/a.txt": ""}, "dirs": ["/"]},
            "final_answer": "done",
        },
    )
    record = check_task(task, impl="stateful", reward_fn=None)
    assert record["ok"] is True
    assert record["goal_satisfied"] is None
    assert record["impl"] == "stateful"


def test_replay_check_cli_and_placeholder_ratio(tmp_path: Any) -> None:
    tasks = [
        _replay_fixture(task_id="rc-pass-1", with_returns=True),
        _replay_fixture(task_id="rc-pass-2", with_returns=False),
        make_task(
            task_id="rc-other",
            task_type="search_qa",
            env_type="search",
            config={"backend": "inmemory", "docs": CORPUS},
        ),
    ]
    path = tmp_path / "tasks.jsonl"
    schema.dump_jsonl(tasks, path)
    report_path = tmp_path / "report.json"
    exit_code = replay_main(
        [
            "--tasks",
            str(path),
            "--out",
            str(report_path),
            "--impl",
            "replay",
            "--reward",
            "none",
            "--task-types",
            "mt_fc",
            "--quiet",
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["failed"] == 0
    assert summary["placeholder_count"] == 1
    assert summary["placeholder_ratio"] == 0.5
    assert report["failures"] == []


def test_replay_check_cli_fails_on_bad_task(tmp_path: Any) -> None:
    task = _replay_fixture(task_id="rc-bad")
    # Remove a schema-required key: the checker must report it, not skip it.
    del task["meta"]["gt_returns_source"]
    path = tmp_path / "bad.jsonl"
    schema.dump_jsonl([task], path)
    report_path = tmp_path / "report.json"
    exit_code = replay_main(
        [
            "--tasks",
            str(path),
            "--out",
            str(report_path),
            "--impl",
            "replay",
            "--reward",
            "none",
            "--quiet",
        ]
    )
    assert exit_code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["failed"] == 1
    assert report["summary"]["passed"] == 0
    assert report["failures"][0]["task_id"] == "rc-bad"
    assert report["failures"][0]["stage"] == "schema"
