"""Local Wikipedia + E5 search environment (plan §1.2, §4.3, §10).

Three layers, all importable without heavy dependencies:

1. **Backends** — :class:`InMemorySearchBackend` (fake index for tests / offline
   smoke runs) and :class:`E5FaissBackend` (Wikipedia passages embedded with an
   E5 model, searched with faiss / numpy; torch + faiss + sentence-transformers
   are imported lazily and guarded).
2. **HTTP service** — :func:`create_app` builds a FastAPI app (guarded import)
   exposing ``POST /search``, ``GET /fetch`` and ``GET /healthz``; the backend
   call runs in a thread pool with a hard server-side timeout so a slow index
   can never hang a rollout.  :class:`SearchClient` is a stdlib-``urllib``
   client with timeout + exponential backoff retries (no ``requests``
   dependency).
3. **Env** — :class:`SearchEnv` implements ``BaseEnv`` with the unified
   ``search(query, top_k)`` / ``fetch(doc_id)`` tool schema and the Search-R1
   cap of **at most 8 rounds per question** (plan §1.2), combined with the
   ``max_turns`` curriculum via ``min``.

Unified tool results (C2): ``search`` returns
``{"results": [{"doc_id", "title", "snippet", "score"}, ...]}`` and ``fetch``
returns ``{"doc_id", "title", "text"}`` — both as canonical JSON in
``tool_results[i].content``.  A backend failure is a recoverable
``ok=False`` result (never an exception into the trainer); the client retries
timeouts / 429 / 5xx twice with exponential backoff.

Index building CLI (user machine, not executed in this session)::

    python -m agentic_rl_data.envs.search build-index \
        --docs data/wiki/e5_docs.jsonl --out-prefix data/wiki/e5_index \
        --model intfloat/e5-base-v2 --batch-size 64 --device cuda:0

Serving / smoke testing / load testing commands live in
``reports/env_setup.md``.  This module is stdlib-only at import time.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..schema import canonical_json
from .base import BaseEnv, EnvError, make_tool_result, truncate_text

DEFAULT_TOP_K = 3
DEFAULT_SNIPPET_CHARS = 1000
DEFAULT_REQUEST_TIMEOUT_S = 5.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_S = 0.5
DEFAULT_MAX_BACKOFF_S = 2.0
#: Total wall-clock budget for backend calls inside one task (0 = disabled).
#: Without this a degraded backend would cost ``timeout × (retries + 1) × 8
#: rounds`` ≈ 130s per task; the deadline turns that into ``ok=False`` results
#: after the budget is spent instead of stalling the rollout.
DEFAULT_TASK_DEADLINE_S = 30.0
#: Search-R1 setting: at most 8 search rounds per question (plan §1.2).
SEARCH_R1_MAX_ROUNDS = 8
#: Environment-variable prefix for the uvicorn factory entry point.
ENV_PREFIX = "ARD_SEARCH_"

#: Transitional backend aliases (CONTRACTS C11 / D12): T1 used to emit
#: ``local_wikipedia_e5`` before the canonical ``e5`` form was agreed.
BACKEND_ALIASES = {
    "local_wikipedia_e5": "e5",
    "local_wikipedia": "e5",
    "wikipedia_e5": "e5",
    "wiki_e5": "e5",
    "local_e5": "e5",
    "e5_faiss": "e5",
    "faiss": "e5",
    "in_memory": "inmemory",
    "local_inmemory": "inmemory",
    "memory": "inmemory",
    "http_service": "http",
    "remote": "http",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalise_search_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a canonical ``env.config`` for :class:`SearchEnv`.

    Backward-compatible with the pre-C11 loader form (D12):

    * ``backend: "local_wikipedia_e5"`` -> ``"e5"`` (aliases in
      :data:`BACKEND_ALIASES`);
    * ``retriever: {"url": ..., "topk": ...}`` -> top-level ``url`` / ``top_k``
      (``topk`` and ``top_k`` both accepted), with any other retriever keys
      (``docs_path`` / ``index_path`` / ``model`` / ``device`` / timeouts)
      promoted to the top level when not already set.

    The canonical form stays ``{backend: e5|inmemory|http, url, top_k,
    max_rounds, retriever}``; this function only widens what we accept.
    """
    normalised = dict(config or {})
    backend = normalised.get("backend")
    if isinstance(backend, str):
        alias = BACKEND_ALIASES.get(backend.strip().lower())
        if alias:
            normalised["backend"] = alias

    retriever = normalised.get("retriever")
    if retriever is not None:
        if isinstance(retriever, str) and not normalised.get("url"):
            if retriever.strip().startswith("http"):
                normalised["url"] = retriever.strip()
        elif isinstance(retriever, Mapping):
            if not normalised.get("url"):
                url = retriever.get("url") or retriever.get("base_url")
                if isinstance(url, str) and url.strip():
                    normalised["url"] = url.strip()
            if normalised.get("top_k") is None:
                top_k = retriever.get("topk", retriever.get("top_k"))
                if top_k is not None:
                    normalised["top_k"] = top_k
            for key in (
                "docs_path",
                "index_path",
                "docs",
                "model",
                "device",
                "snippet_chars",
                "request_timeout_s",
                "max_retries",
                "backoff_s",
                "max_backoff_s",
                "max_rounds",
                "task_deadline_s",
                "max_missing_ratio",
            ):
                if normalised.get(key) is None and retriever.get(key) is not None:
                    normalised[key] = retriever[key]
            if normalised.get("backend") is None:
                hint = retriever.get("backend") or retriever.get("type")
                if isinstance(hint, str) and hint.strip():
                    normalised["backend"] = BACKEND_ALIASES.get(
                        hint.strip().lower(), hint.strip().lower()
                    )

    # A loader that says "local wikipedia + E5" and gives a service URL means
    # the E5 index lives behind that HTTP service, not in this process (D12).
    backend = normalised.get("backend")
    if (
        isinstance(backend, str)
        and backend.strip().lower() in ("e5", "local_wikipedia_e5", "wikipedia_e5")
        and normalised.get("url")
        and not (normalised.get("docs_path") or normalised.get("index_path")
                 or normalised.get("index") or normalised.get("docs"))
    ):
        normalised["backend"] = "http"
    return normalised


class SearchBackendError(RuntimeError):
    """Search backend / transport failure (recoverable by the env).

    ``retryable`` tells :class:`SearchClient` whether another attempt may help;
    the env converts any escaping error into a failed tool result.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status = status


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def _tokens(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def make_snippet(text: Any, query: str = "", max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    """Build a deterministic query-centred snippet."""
    if not isinstance(text, str):
        text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    lowered = text.lower()
    position = -1
    for token in _tokens(query):
        index = lowered.find(token)
        if index >= 0:
            position = index
            break
    if position < 0:
        return truncate_text(text, max_chars)[0]
    start = max(0, position - max(100, max_chars // 4))
    end = min(len(text), start + max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return truncate_text(snippet, max_chars)[0]


def _document_id(entry: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("doc_id", "docid", "id", "url"):
        value = entry.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return f"doc_{fallback_index}"


def _document_title(entry: Mapping[str, Any], doc_id: str) -> str:
    title = entry.get("title") or entry.get("heading")
    return title.strip() if isinstance(title, str) and title.strip() else doc_id


def _document_text(entry: Mapping[str, Any]) -> str:
    for key in ("text", "contents", "body", "passage", "content"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return ""


def load_documents(path: str | Path) -> list[dict[str, Any]]:
    """Load a docs JSONL file into ``[{"doc_id","title","text"}]``."""
    docs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SearchBackendError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(entry, Mapping):
                continue
            doc_id = _document_id(entry, len(docs))
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": _document_title(entry, doc_id),
                    "text": _document_text(entry),
                }
            )
    return docs


class InMemorySearchBackend:
    """Dependency-free BM25-lite index over an in-memory document list.

    Deterministic: ties are broken by ``doc_id``.  Empty query / no token
    overlap returns ``[]`` (Search-R1 behaviour: the model must reformulate).
    """

    name = "inmemory"

    def __init__(
        self,
        docs: Iterable[Mapping[str, Any]],
        *,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> None:
        self.snippet_chars = int(snippet_chars)
        self._docs: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        # term -> {doc_index: term frequency}; title tokens weigh 3x.
        self._postings: dict[str, dict[int, int]] = {}
        for index, entry in enumerate(docs):
            if not isinstance(entry, Mapping):
                continue
            doc_id = _document_id(entry, len(self._docs))
            title = _document_title(entry, doc_id)
            text = _document_text(entry)
            self._docs.append({"doc_id": doc_id, "title": title, "text": text})
            self._by_id.setdefault(doc_id, self._docs[-1])
            frequencies: dict[str, int] = {}
            for token in _tokens(title):
                frequencies[token] = frequencies.get(token, 0) + 3
            for token in _tokens(text):
                frequencies[token] = frequencies.get(token, 0) + 1
            doc_index = len(self._docs) - 1
            for token, frequency in frequencies.items():
                self._postings.setdefault(token, {})[doc_index] = frequency

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> "InMemorySearchBackend":
        return cls(load_documents(path), snippet_chars=snippet_chars)

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        top_k = max(1, int(top_k))
        query_terms = sorted(set(_tokens(query)))
        if not query_terms:
            return []
        count = max(1, len(self._docs))
        scores: dict[int, float] = {}
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            idf = math.log(1.0 + count / (1.0 + document_frequency))
            for doc_index, frequency in postings.items():
                # BM25-lite saturation (k1 = 1.2, no document-length norm).
                saturation = (frequency * 2.2) / (frequency + 1.2)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * saturation
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: list[dict[str, Any]] = []
        for doc_index, score in ranked:
            doc = self._docs[doc_index]
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "snippet": make_snippet(doc["text"], query, self.snippet_chars),
                    "score": round(float(score), 6),
                }
            )
        return results

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        key = str(doc_id)
        doc = self._by_id.get(key)
        if doc is None:
            raise SearchBackendError(f"document not found: {key!r}", retryable=False)
        return {"doc_id": doc["doc_id"], "title": doc["title"], "text": doc["text"]}

    def close(self) -> None:
        return None


def _load_docs_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return load_documents(path)


class E5FaissBackend:
    """E5 embeddings + faiss (or numpy fallback) over Wikipedia passages.

    Heavy imports are lazy: constructing the object only reads JSONL docs and
    an index file; ``faiss`` / ``numpy`` are imported by :meth:`_load_index`,
    and ``sentence-transformers`` when the first query is encoded.  Missing
    dependencies surface as :class:`SearchBackendError` with the pip command,
    never as a raw ImportError into the rollout loop.
    """

    name = "e5"

    def __init__(
        self,
        docs_path: str | Path,
        index_path: str | Path,
        *,
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
        normalize: bool = True,
    ) -> None:
        self._docs = _load_docs_jsonl(docs_path)
        self._doc_by_id: dict[str, dict[str, Any]] = {}
        for doc in self._docs:
            self._doc_by_id.setdefault(doc["doc_id"], doc)
        self._index_path = Path(index_path)
        self._model_name = model_name
        self._device = device
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self.snippet_chars = int(snippet_chars)
        self._normalize = bool(normalize)
        self._lock = threading.Lock()
        self._model: Any = None
        self._faiss: Any = None
        self._index: Any = None
        self._vectors: Any = None
        self._load_index()

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def _load_index(self) -> None:
        path = self._index_path
        if not path.exists():
            raise SearchBackendError(f"index file not found: {path}", retryable=False)
        suffix = path.suffix.lower()
        if suffix == ".faiss":
            try:
                import faiss  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SearchBackendError(
                    "loading a .faiss index requires faiss-cpu / faiss-gpu: "
                    "pip install faiss-cpu",
                    retryable=False,
                ) from exc
            self._faiss = faiss
            self._index = faiss.read_index(str(path))
            return
        if suffix in (".npz", ".npy"):
            try:
                import numpy as np  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SearchBackendError(
                    "loading an .npz index requires numpy: pip install numpy",
                    retryable=False,
                ) from exc
            payload = np.load(str(path))
            try:
                vectors = payload["vectors"] if "vectors" in payload else payload["arr_0"]
            finally:
                payload.close()
            self._vectors = np.asarray(vectors, dtype="float32")
            if self._normalize:
                norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                self._vectors = self._vectors / norms
            return
        raise SearchBackendError(
            f"unsupported index format {suffix!r}; expected .faiss or .npz",
            retryable=False,
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SearchBackendError(
                "E5 retrieval requires sentence-transformers: "
                "pip install sentence-transformers",
                retryable=False,
            ) from exc
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:  # pragma: no cover - model / device errors
            raise SearchBackendError(
                f"could not load E5 model {self._model_name!r}: {exc}", retryable=False
            ) from exc
        return self._model

    def _encode_query(self, query: str) -> Any:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SearchBackendError(
                "E5 retrieval requires numpy: pip install numpy", retryable=False
            ) from exc
        model = self._ensure_model()
        with self._lock:
            vector = model.encode(
                [self._query_prefix + query],
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vector, dtype="float32")[0]

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        top_k = max(1, int(top_k))
        vector = self._encode_query(query)
        if self._index is not None and self._faiss is not None:
            scores, indices = self._index.search(vector.reshape(1, -1), top_k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            try:
                import numpy as np  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise SearchBackendError("numpy is required for the .npz index", retryable=False) from exc
            if self._vectors is None or self._vectors.shape[0] == 0:
                return []
            scores = self._vectors @ vector
            order = np.argsort(-scores)[:top_k]
            pairs = [(int(index), float(scores[index])) for index in order]
        results: list[dict[str, Any]] = []
        for index, score in pairs:
            if index is None or int(index) < 0 or int(index) >= len(self._docs):
                continue
            doc = self._docs[int(index)]
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "snippet": make_snippet(doc["text"], query, self.snippet_chars),
                    "score": round(float(score), 6),
                }
            )
        return results

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        key = str(doc_id)
        doc = self._doc_by_id.get(key)
        if doc is None:
            raise SearchBackendError(f"document not found: {key!r}", retryable=False)
        return {"doc_id": doc["doc_id"], "title": doc["title"], "text": doc["text"]}

    def close(self) -> None:
        self._model = None
        self._index = None
        self._vectors = None
        self._faiss = None


# ---------------------------------------------------------------------------
# index builder (E5)
# ---------------------------------------------------------------------------


def build_e5_index(
    docs_path: str | Path,
    out_prefix: str | Path,
    *,
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 64,
    device: str | None = None,
    passage_prefix: str = "passage: ",
    max_docs: int | None = None,
    write_faiss: bool = True,
) -> dict[str, Any]:
    """Encode docs JSONL with E5 and write ``<prefix>.npz`` (+ ``.faiss``).

    Also writes ``<prefix>.docs.jsonl`` (normalised doc records) and
    ``<prefix>.meta.json`` so the server can be restarted without re-encoding.
    Production-scale corpora should use a faiss IVF+PQ index built offline
    (see ``reports/env_setup.md``); this builder writes a flat index and is
    intended for sub-million-passage smoke runs.
    """
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SearchBackendError(
            "build-index requires numpy: pip install numpy", retryable=False
        ) from exc
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SearchBackendError(
            "build-index requires sentence-transformers: pip install sentence-transformers",
            retryable=False,
        ) from exc

    docs = _load_docs_jsonl(docs_path)
    if max_docs is not None:
        docs = docs[: int(max_docs)]
    if not docs:
        raise SearchBackendError(f"no documents found in {docs_path}", retryable=False)

    model = SentenceTransformer(model_name, device=device)
    vectors: list[Any] = []
    for start in range(0, len(docs), max(1, int(batch_size))):
        batch = docs[start : start + max(1, int(batch_size))]
        texts = [passage_prefix + doc["text"] for doc in batch]
        encoded = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors.append(np.asarray(encoded, dtype="float32"))
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = Path(str(out_prefix) + ".npz")
    np.savez(str(npz_path), vectors=matrix)

    docs_path_out = Path(str(out_prefix) + ".docs.jsonl")
    with docs_path_out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(canonical_json(doc) + "\n")

    meta = {
        "model": model_name,
        "count": len(docs),
        "dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "passage_prefix": passage_prefix,
        "normalized": True,
        "source_docs": str(docs_path),
        "npz": str(npz_path),
        "docs_jsonl": str(docs_path_out),
    }

    faiss_written = False
    if write_faiss and len(docs) > 0:
        try:
            import faiss  # type: ignore
        except ImportError:
            faiss = None  # type: ignore
        if faiss is not None:
            index = faiss.IndexFlatIP(int(matrix.shape[1]))
            index.add(matrix)
            faiss_path = Path(str(out_prefix) + ".faiss")
            faiss.write_index(index, str(faiss_path))
            meta["faiss"] = str(faiss_path)
            faiss_written = True

    with Path(str(out_prefix) + ".meta.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2))

    return {
        "count": len(docs),
        "dim": meta["dim"],
        "npz": str(npz_path),
        "docs_jsonl": str(docs_path_out),
        "faiss": meta.get("faiss"),
        "faiss_written": faiss_written,
    }


# ---------------------------------------------------------------------------
# HTTP client (stdlib urllib, timeout + exponential backoff)
# ---------------------------------------------------------------------------


def _http_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> Any:
    """Default transport: one JSON request via :mod:`urllib`, no retries."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        retryable = exc.code in (408, 429, 500, 502, 503, 504)
        body = b""
        try:
            body = exc.read()[:200]
        except Exception:  # pragma: no cover - defensive
            body = b""
        raise SearchBackendError(
            f"HTTP {exc.code} from {url}: {body.decode('utf-8', 'replace')}",
            retryable=retryable,
            status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise SearchBackendError(
            f"network error for {url}: {exc.reason}", retryable=True
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SearchBackendError(f"timeout after {timeout}s for {url}", retryable=True) from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SearchBackendError(f"malformed JSON from {url}: {exc}", retryable=False) from exc


Transport = Callable[[str, str, "Mapping[str, Any] | None", float], Any]


class SearchClient:
    """Thread-safe HTTP client for the search service.

    ``transport`` is injectable so tests can exercise retry / timeout logic
    without a live server.  The default transport is stdlib ``urllib``.
    """

    name = "http"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise SearchBackendError("base_url must be a non-empty string")
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self.max_backoff_s = max(0.0, float(max_backoff_s))
        self._transport: Transport = transport or _http_json

    @property
    def doc_count(self) -> None:
        return None

    def _endpoint(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        last_error: SearchBackendError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            if attempt > 0:
                delay = min(self.backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
                if delay > 0:
                    time.sleep(delay)
            try:
                return self._transport(
                    method, self._endpoint(path), payload, self.timeout_s
                )
            except SearchBackendError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
            except Exception as exc:  # defensive: transport contract violation
                last_error = SearchBackendError(str(exc), retryable=True)
        raise SearchBackendError(
            f"{method} {path}: retries exhausted after {attempts} attempt(s): {last_error}",
            retryable=False,
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise SearchBackendError("search query must be a non-empty string")
        payload = {"query": query, "top_k": max(1, int(top_k))}
        response = self._request("POST", "/search", payload)
        results = response.get("results") if isinstance(response, Mapping) else None
        if not isinstance(results, list):
            raise SearchBackendError("malformed /search response: missing 'results' list")
        normalised: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            normalised.append(
                {
                    "doc_id": item.get("doc_id"),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", item.get("text", "")),
                    "score": item.get("score", 0.0),
                }
            )
        return normalised

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        path = "/fetch?doc_id=" + urllib.parse.quote(str(doc_id), safe="")
        response = self._request("GET", path)
        if not isinstance(response, Mapping):
            raise SearchBackendError("malformed /fetch response")
        return {
            "doc_id": response.get("doc_id", doc_id),
            "title": response.get("title", ""),
            "text": response.get("text", ""),
        }

    def healthz(self) -> dict[str, Any]:
        response = self._request("GET", "/healthz")
        return dict(response) if isinstance(response, Mapping) else {"status": "unknown"}

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# FastAPI service (optional dependency, guarded)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard only
    import fastapi as _fastapi_import_probe  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency absent
    FASTAPI_AVAILABLE = False
else:  # pragma: no cover - optional dependency present
    FASTAPI_AVAILABLE = True


def _require_fastapi() -> None:
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "the search HTTP service requires fastapi and uvicorn: "
            "pip install fastapi uvicorn"
        )


def create_app(
    backend: Any,
    *,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    max_workers: int = 32,
) -> Any:
    """Build the FastAPI app around ``backend``.

    ``POST /search``  body ``{"query": str, "top_k": int}``
    ``GET  /fetch``   query ``doc_id``
    ``GET  /healthz`` service health + backend metadata

    Backend calls run in a shared thread pool with a hard timeout; a timeout or
    backend error becomes HTTP 503 (rollout clients retry / degrade).
    """
    _require_fastapi()
    from fastapi import FastAPI, HTTPException, Query  # type: ignore
    from pydantic import BaseModel  # type: ignore

    app = FastAPI(title="agentic_rl_data search server", version="1.0")
    executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
    app.state.backend = backend
    app.state.executor = executor

    class SearchRequest(BaseModel):
        query: str
        top_k: int = DEFAULT_TOP_K

    def _run(function: Callable[[], Any]) -> Any:
        future = executor.submit(function)
        try:
            return future.result(timeout=max(0.1, float(request_timeout_s)))
        except (TimeoutError, FuturesTimeoutError) as exc:
            future.cancel()
            raise SearchBackendError(
                f"search backend timed out after {request_timeout_s}s", retryable=False
            ) from exc
        except SearchBackendError:
            raise
        except Exception as exc:
            raise SearchBackendError(
                f"search backend call failed: {exc}", retryable=True
            ) from exc

    @app.post("/search")
    def search_endpoint(request: SearchRequest) -> dict[str, Any]:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query must be non-empty")
        try:
            results = _run(lambda: backend.search(request.query, request.top_k))
        except SearchBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"results": results}

    @app.get("/fetch")
    def fetch_endpoint(doc_id: str = Query(..., description="document id")) -> dict[str, Any]:
        try:
            document = _run(lambda: backend.fetch(doc_id))
        except SearchBackendError as exc:
            status = 404 if "not found" in str(exc).lower() else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return {"doc_id": document.get("doc_id"), "title": document.get("title"), "text": document.get("text")}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": getattr(backend, "name", type(backend).__name__),
            "doc_count": getattr(backend, "doc_count", None),
        }

    return app


# ---------------------------------------------------------------------------
# backend resolution (service + env)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def backend_from_env(environ: Mapping[str, str] | None = None) -> Any:
    """Build the backend from ``ARD_SEARCH_*`` env vars (uvicorn factory)."""
    env = dict(environ if environ is not None else os.environ)
    kind = str(env.get(ENV_PREFIX + "BACKEND", "inmemory")).strip().lower()
    snippet_chars = int(env.get(ENV_PREFIX + "SNIPPET_CHARS", DEFAULT_SNIPPET_CHARS))
    if kind == "inmemory":
        docs_path = env.get(ENV_PREFIX + "DOCS")
        if not docs_path:
            raise SearchBackendError(
                "backend=inmemory requires ARD_SEARCH_DOCS pointing at a docs JSONL"
            )
        return InMemorySearchBackend.from_jsonl(docs_path, snippet_chars=snippet_chars)
    if kind in ("e5", "faiss"):
        docs_path = env.get(ENV_PREFIX + "DOCS")
        index_path = env.get(ENV_PREFIX + "INDEX")
        if not docs_path or not index_path:
            raise SearchBackendError(
                "backend=e5 requires ARD_SEARCH_DOCS and ARD_SEARCH_INDEX"
            )
        return E5FaissBackend(
            docs_path,
            index_path,
            model_name=env.get(ENV_PREFIX + "MODEL", "intfloat/e5-base-v2"),
            device=env.get(ENV_PREFIX + "DEVICE") or None,
            snippet_chars=snippet_chars,
        )
    raise SearchBackendError(f"unknown ARD_SEARCH_BACKEND: {kind!r}")


def app() -> Any:
    """uvicorn factory: ``uvicorn agentic_rl_data.envs.search:app --factory``."""
    return create_app(backend_from_env(), request_timeout_s=_env_float("REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_S))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def serve(
    host: str = "0.0.0.0",
    port: int = 8899,
    *,
    backend: Any | None = None,
    workers: int = 1,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    max_workers: int = 32,
) -> None:
    """Start the search service (blocking).  Uses the backend from env vars.

    ``workers > 1`` delegates to uvicorn's process pool via the module factory
    (each worker loads the index — memory multiplies; see env_setup.md).
    """
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "serving requires uvicorn: pip install uvicorn fastapi"
        ) from exc
    worker_count = max(1, int(workers))
    if worker_count > 1:
        uvicorn.run(
            "agentic_rl_data.envs.search:app",
            factory=True,
            host=host,
            port=int(port),
            workers=worker_count,
            log_level="info",
        )
        return
    if backend is None:
        backend = backend_from_env()
    application = create_app(
        backend, request_timeout_s=request_timeout_s, max_workers=max_workers
    )
    uvicorn.run(application, host=host, port=int(port), log_level="info")


# ---------------------------------------------------------------------------
# SearchEnv
# ---------------------------------------------------------------------------


def _argument(arguments: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in arguments:
            return arguments[name]
    return default


def _coerce_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_deadline(value: Any) -> float:
    """Task deadline in seconds; ``<= 0`` (or invalid) disables it."""
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


class SearchEnv(BaseEnv):
    """Search environment: ``search(query, top_k)`` + ``fetch(doc_id)`` tools.

    ``env.config`` keys (canonical form, C11)::

        {"backend": "inmemory" | "http" | "e5" | backend object,
         "retriever": {"url": ..., "topk": ...},  # transitional alias (D12)
         "docs": [...] | "docs_path": "...",       # inmemory
         "url": "http://127.0.0.1:8899",           # http
         "docs_path"/"index_path"/"model"/"device": # e5
         "top_k": 3, "max_rounds": 8,
         "task_deadline_s": 30,
         "search_tools": [...], "fetch_tools": [...],
         "snippet_chars": 1000,
         "request_timeout_s": 5, "max_retries": 2,
         "own_backend": bool}

    Turn cap = ``min(task.meta.max_turns, reset override, config.max_rounds,
    8)`` — Search-R1's hard limit.  A **task-level deadline** (default 30s,
    ``task_deadline_s <= 0`` disables) bounds total backend wall-clock time:
    once spent, further tool calls return ``ok=False`` and
    ``info["backend_error"]=True`` instead of stalling the rollout.  Backend
    failures are always recoverable ``ok=False`` results; no exception escapes
    :meth:`step`.
    """

    env_type = "search"
    impl = "local"

    def _setup(self, task: Mapping[str, Any], config: dict[str, Any]) -> None:
        config = normalise_search_config(config)
        self._config = config
        self._top_k = _coerce_positive_int(config.get("top_k"), DEFAULT_TOP_K)
        self._snippet_chars = _coerce_positive_int(
            config.get("snippet_chars"), DEFAULT_SNIPPET_CHARS
        )
        self._max_rounds = min(
            _coerce_positive_int(config.get("max_rounds"), SEARCH_R1_MAX_ROUNDS),
            SEARCH_R1_MAX_ROUNDS,
        )
        self._search_tools = {
            str(name)
            for name in (config.get("search_tools") or ("search", "web_search", "search_wiki"))
        }
        self._fetch_tools = {
            str(name)
            for name in (config.get("fetch_tools") or ("fetch", "fetch_page", "fetch_doc"))
        }
        self._task_deadline_s = _coerce_deadline(
            config.get("task_deadline_s", DEFAULT_TASK_DEADLINE_S)
        )
        self._started_at = time.monotonic()
        self._deadline_exceeded = False
        try:
            self._backend, self._owns_backend = _resolve_backend(
                config, deadline_s=self._task_deadline_s
            )
        except EnvError:
            raise
        except Exception as exc:  # optional deps / index files: degrade, don't crash
            self._backend = _UnavailableBackend(
                f"{type(exc).__name__}: {exc}"
            )
            self._owns_backend = True
        self._queries: list[dict[str, Any]] = []
        self._fetched: list[str] = []
        self._backend_errors: int = 0
        self._backend_error_flag: bool = False

    def _deadline_remaining(self) -> float | None:
        if self._task_deadline_s <= 0:
            return None
        return self._task_deadline_s - (time.monotonic() - self._started_at)

    def _deadline_hit(self) -> bool:
        remaining = self._deadline_remaining()
        return remaining is not None and remaining <= 0.0

    def _turn_cap(
        self,
        task: Mapping[str, Any],
        config: dict[str, Any],
    ) -> int | None:
        config = normalise_search_config(config)
        return min(
            _coerce_positive_int(config.get("max_rounds"), SEARCH_R1_MAX_ROUNDS),
            SEARCH_R1_MAX_ROUNDS,
        )

    def _handle_tool_calls(
        self,
        calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        for call in calls:
            tool = _match_tool_alias(call["name"], self._search_tools, self._fetch_tools)
            if tool == "search":
                results.append(self._search_call(call))
            elif tool == "fetch":
                results.append(self._fetch_call(call))
            else:
                results.append(
                    make_tool_result(
                        call["name"],
                        f"Error: unknown search tool {call['name']!r}; "
                        f"available: {sorted(self._search_tools | self._fetch_tools)}",
                        ok=False,
                        call_id=call["id"],
                    )
                )
        info["search"] = {
            "queries": list(self._queries),
            "fetched_docs": list(self._fetched),
            "backend_errors": self._backend_errors,
        }
        if self._backend_error_flag:
            info["backend_error"] = True
        if self._deadline_exceeded:
            info["deadline_exceeded"] = True
        return results, info

    def _search_call(self, call: dict[str, Any]) -> dict[str, Any]:
        query = _argument(call["arguments"], "query", "q", "search_query", "keywords")
        if not isinstance(query, str) or not query.strip():
            return make_tool_result(
                call["name"],
                "Error: search requires a non-empty 'query' string",
                ok=False,
                call_id=call["id"],
            )
        top_k = _coerce_positive_int(
            _argument(call["arguments"], "top_k", "k", "num_results", default=self._top_k),
            self._top_k,
        )
        if self._deadline_hit():
            return self._deadline_result(
                call, {"query": query, "top_k": top_k}, tool="search"
            )
        record: dict[str, Any] = {"query": query, "top_k": top_k}
        try:
            results = self._backend.search(query, top_k)
        except Exception as exc:  # backend must never crash the rollout
            self._backend_errors += 1
            self._backend_error_flag = True
            record["ok"] = False
            self._queries.append(record)
            return make_tool_result(
                call["name"],
                f"Error: search backend unavailable: {type(exc).__name__}: {exc}",
                ok=False,
                call_id=call["id"],
            )
        normalised = [
            {
                "doc_id": item.get("doc_id"),
                "title": item.get("title", ""),
                "snippet": truncate_text(item.get("snippet", ""), self._snippet_chars)[0],
                "score": item.get("score", 0.0),
            }
            for item in results
        ]
        record["ok"] = True
        record["num_results"] = len(normalised)
        self._queries.append(record)
        return make_tool_result(
            call["name"],
            canonical_json({"results": normalised}),
            ok=True,
            call_id=call["id"],
        )

    def _fetch_call(self, call: dict[str, Any]) -> dict[str, Any]:
        doc_id = _argument(call["arguments"], "doc_id", "id", "url", "document_id")
        if not isinstance(doc_id, (str, int)) or not str(doc_id).strip():
            return make_tool_result(
                call["name"],
                "Error: fetch requires a non-empty 'doc_id'",
                ok=False,
                call_id=call["id"],
            )
        if self._deadline_hit():
            return self._deadline_result(call, {"doc_id": str(doc_id)}, tool="fetch")
        try:
            document = self._backend.fetch(doc_id)
        except Exception as exc:
            self._backend_errors += 1
            self._backend_error_flag = True
            return make_tool_result(
                call["name"],
                f"Error: fetch backend unavailable: {type(exc).__name__}: {exc}",
                ok=False,
                call_id=call["id"],
            )
        self._fetched.append(str(document.get("doc_id", doc_id)))
        return make_tool_result(
            call["name"],
            canonical_json(
                {
                    "doc_id": document.get("doc_id"),
                    "title": document.get("title", ""),
                    "text": truncate_text(
                        document.get("text", ""), self.max_result_chars
                    )[0],
                }
            ),
            ok=True,
            call_id=call["id"],
        )

    def _deadline_result(
        self,
        call: dict[str, Any],
        record: dict[str, Any],
        *,
        tool: str,
    ) -> dict[str, Any]:
        """Return a recoverable failure once the task-level deadline is spent."""
        self._deadline_exceeded = True
        self._backend_error_flag = True
        self._backend_errors += 1
        record["ok"] = False
        record["deadline_exceeded"] = True
        record["tool"] = tool
        self._queries.append(record)
        return make_tool_result(
            call["name"],
            f"Error: search task deadline of {self._task_deadline_s}s exceeded; "
            f"{tool} backend call skipped",
            ok=False,
            call_id=call["id"],
        )

    def _env_state_fields(self) -> dict[str, Any]:
        elapsed = round(time.monotonic() - self._started_at, 3)
        return {
            "num_turns": len(self._turns),
            "timeout": bool(self._timeout),
            "rendered": (
                f"search backend={getattr(self._backend, 'name', type(self._backend).__name__)} "
                f"queries={len(self._queries)} fetched={len(self._fetched)} "
                f"errors={self._backend_errors} elapsed={elapsed}s"
            ),
            "extras": {
                "backend": getattr(self._backend, "name", type(self._backend).__name__),
                "doc_count": getattr(self._backend, "doc_count", None),
                "top_k": self._top_k,
                "max_rounds": self._max_rounds,
                "task_deadline_s": self._task_deadline_s,
                "elapsed_s": elapsed,
                "deadline_exceeded": self._deadline_exceeded,
                "queries": list(self._queries),
                "fetched_docs": list(self._fetched),
                "backend_errors": self._backend_errors,
                "snippet_chars": self._snippet_chars,
                "retriever_aliased": bool(self._config.get("retriever")),
            },
        }

    def _teardown(self) -> None:
        if self._owns_backend:
            try:
                self._backend.close()
            except Exception:
                pass


def _match_tool_alias(name: str, search_tools: set[str], fetch_tools: set[str]) -> str | None:
    candidate = str(name).strip()
    if candidate in search_tools:
        return "search"
    if candidate in fetch_tools:
        return "fetch"
    lowered = candidate.lower()
    if lowered in {item.lower() for item in search_tools}:
        return "search"
    if lowered in {item.lower() for item in fetch_tools}:
        return "fetch"
    if "fetch" in lowered or "document" in lowered:
        return "fetch"
    if "search" in lowered or "wiki" in lowered:
        return "search"
    return None


def _resolve_backend(
    config: dict[str, Any],
    *,
    deadline_s: float = 0.0,
) -> tuple[Any, bool]:
    """Resolve ``env.config`` into a backend; returns ``(backend, owns)``.

    ``deadline_s`` (task budget, seconds; 0 = disabled) clamps the HTTP client
    timeout so a single call cannot overshoot the remaining budget by much.
    """
    config = normalise_search_config(config)
    raw = config.get("backend")
    if raw is None:
        raw = config.get("backend_obj")
    if raw is not None and not isinstance(raw, str):
        if not (callable(getattr(raw, "search", None)) and callable(getattr(raw, "fetch", None))):
            raise EnvError(
                "env.config.backend must be 'inmemory' | 'http' | 'e5' or a backend object"
            )
        return raw, bool(config.get("own_backend", False))

    if isinstance(raw, str):
        kind = raw.strip().lower()
    elif config.get("url") or config.get("base_url"):
        kind = "http"
    elif config.get("docs") or config.get("docs_path"):
        kind = "inmemory"
    elif config.get("index_path") or config.get("index"):
        kind = "e5"
    else:
        raise EnvError(
            "SearchEnv requires env.config.backend "
            "('inmemory' | 'http' | 'e5') with its backend options"
        )

    snippet_chars = _coerce_positive_int(config.get("snippet_chars"), DEFAULT_SNIPPET_CHARS)

    if kind == "inmemory":
        docs = config.get("docs")
        if isinstance(docs, str):
            return InMemorySearchBackend.from_jsonl(
                docs, snippet_chars=snippet_chars
            ), True
        if docs is None:
            docs_path = config.get("docs_path")
            if not docs_path:
                raise EnvError("backend 'inmemory' requires env.config.docs or env.config.docs_path")
            return InMemorySearchBackend.from_jsonl(
                docs_path, snippet_chars=snippet_chars
            ), True
        if not isinstance(docs, (list, tuple)):
            raise EnvError("env.config.docs must be a list of document mappings")
        return InMemorySearchBackend(docs, snippet_chars=snippet_chars), True

    if kind == "http":
        base_url = config.get("url") or config.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise EnvError("backend 'http' requires env.config.url")
        request_timeout = float(config.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S))
        if deadline_s > 0:
            request_timeout = min(request_timeout, float(deadline_s))
        client = SearchClient(
            base_url,
            timeout_s=request_timeout,
            max_retries=int(config.get("max_retries", DEFAULT_MAX_RETRIES)),
            backoff_s=float(config.get("backoff_s", DEFAULT_BACKOFF_S)),
            max_backoff_s=float(config.get("max_backoff_s", DEFAULT_MAX_BACKOFF_S)),
        )
        return client, True

    if kind == "e5":
        docs_path = config.get("docs_path") or config.get("docs")
        index_path = config.get("index_path") or config.get("index")
        if not isinstance(docs_path, str) or not isinstance(index_path, str):
            # Loader-produced tasks carry no local index paths (they are runtime
            # settings): degrade to a recoverable backend so env construction
            # never fails and the first tool call reports backend_error (C11).
            return _UnavailableBackend(
                "backend 'e5' requires env.config.docs_path and env.config.index_path "
                "(or an http url / ARD_SEARCH_* env vars)"
            ), True
        try:
            backend = E5FaissBackend(
                docs_path,
                index_path,
                model_name=str(config.get("model", "intfloat/e5-base-v2")),
                device=config.get("device") or None,
                snippet_chars=snippet_chars,
            )
        except SearchBackendError as exc:
            # Constructing the env must not fail on missing optional deps;
            # the first tool call surfaces it as a recoverable ok=False result.
            return _UnavailableBackend(str(exc)), True
        return backend, True

    raise EnvError(f"unknown search backend {kind!r}")


class _UnavailableBackend:
    """Placeholder backend: every call fails recoverably with a clear reason."""

    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def doc_count(self) -> int:
        return 0

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        raise SearchBackendError(self.reason, retryable=False)

    def fetch(self, doc_id: Any) -> dict[str, Any]:
        raise SearchBackendError(self.reason, retryable=False)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# load test (bench) — user machine, not executed in this session
# ---------------------------------------------------------------------------


def bench_search(
    base_url: str,
    *,
    concurrency: int = 32,
    requests: int = 500,
    top_k: int = DEFAULT_TOP_K,
    query: str = "what is the capital of France",
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    max_retries: int = 0,
) -> dict[str, Any]:
    """Fire ``requests`` searches with ``concurrency`` workers; timing summary.

    Returns ``{"requests", "errors", "qps", "mean_ms", "p50_ms", "p95_ms",
    "p99_ms", "wall_s"}``.  Used by the user-machine load test documented in
    ``reports/env_setup.md`` (not executed here).
    """
    import statistics

    client = SearchClient(
        base_url, timeout_s=timeout_s, max_retries=max_retries
    )

    def _one(_: int) -> tuple[bool, float]:
        start = time.perf_counter()
        try:
            client.search(query, top_k)
            return True, (time.perf_counter() - start) * 1000.0
        except Exception:
            return False, (time.perf_counter() - start) * 1000.0

    started = time.perf_counter()
    latencies: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        for ok, latency in pool.map(_one, range(max(1, int(requests)))):
            latencies.append(latency)
            if not ok:
                errors += 1
    wall = time.perf_counter() - started

    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round(ordered[index], 2)

    return {
        "requests": len(latencies),
        "errors": errors,
        "qps": round(len(latencies) / wall, 2) if wall > 0 else 0.0,
        "mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
        "wall_s": round(wall, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``serve`` / ``search`` / ``fetch`` /
    ``build-index`` / ``bench`` / ``health``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agentic_rl_data.envs.search",
        description="Local Wikipedia + E5 search service and SearchEnv utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="start the HTTP search service")
    serve_parser.add_argument("--backend", default="inmemory", choices=("inmemory", "e5"))
    serve_parser.add_argument("--docs", default=None, help="docs JSONL path")
    serve_parser.add_argument("--index", default=None, help="E5 index (.faiss/.npz)")
    serve_parser.add_argument("--model", default="intfloat/e5-base-v2")
    serve_parser.add_argument("--device", default=None)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8899)
    serve_parser.add_argument("--workers", type=int, default=1)
    serve_parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    serve_parser.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)

    client_parser = subparsers.add_parser("search", help="one search query (smoke)")
    client_parser.add_argument("query")
    client_parser.add_argument("--url", default="http://127.0.0.1:8899")
    client_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    fetch_parser = subparsers.add_parser("fetch", help="one fetch call (smoke)")
    fetch_parser.add_argument("doc_id")
    fetch_parser.add_argument("--url", default="http://127.0.0.1:8899")

    health_parser = subparsers.add_parser("health", help="service health (smoke)")
    health_parser.add_argument("--url", default="http://127.0.0.1:8899")

    index_parser = subparsers.add_parser("build-index", help="build an E5 index")
    index_parser.add_argument("--docs", required=True)
    index_parser.add_argument("--out-prefix", required=True)
    index_parser.add_argument("--model", default="intfloat/e5-base-v2")
    index_parser.add_argument("--batch-size", type=int, default=64)
    index_parser.add_argument("--device", default=None)
    index_parser.add_argument("--max-docs", type=int, default=None)
    index_parser.add_argument("--no-faiss", action="store_true")

    bench_parser = subparsers.add_parser("bench", help="load test the search service")
    bench_parser.add_argument("--url", default="http://127.0.0.1:8899")
    bench_parser.add_argument("--concurrency", type=int, default=32)
    bench_parser.add_argument("--requests", type=int, default=500)
    bench_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    bench_parser.add_argument("--query", default="what is the capital of France")
    bench_parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    bench_parser.add_argument("--retries", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "serve":
        os.environ[ENV_PREFIX + "BACKEND"] = args.backend
        if args.docs:
            os.environ[ENV_PREFIX + "DOCS"] = args.docs
        if args.index:
            os.environ[ENV_PREFIX + "INDEX"] = args.index
        if args.model:
            os.environ[ENV_PREFIX + "MODEL"] = args.model
        if args.device:
            os.environ[ENV_PREFIX + "DEVICE"] = args.device
        os.environ[ENV_PREFIX + "SNIPPET_CHARS"] = str(args.snippet_chars)
        os.environ[ENV_PREFIX + "REQUEST_TIMEOUT"] = str(args.request_timeout)
        print(
            f"[search] serving backend={args.backend} host={args.host} port={args.port} "
            f"workers={args.workers} request_timeout={args.request_timeout}s"
        )
        serve(
            host=args.host,
            port=args.port,
            workers=args.workers,
            request_timeout_s=args.request_timeout,
        )
        return 0

    if args.command in ("search", "fetch", "health"):
        client = SearchClient(args.url)
        if args.command == "search":
            print(canonical_json(client.search(args.query, args.top_k)))
        elif args.command == "fetch":
            print(canonical_json(client.fetch(args.doc_id)))
        else:
            print(canonical_json(client.healthz()))
        return 0

    if args.command == "build-index":
        summary = build_e5_index(
            args.docs,
            args.out_prefix,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            max_docs=args.max_docs,
            write_faiss=not args.no_faiss,
        )
        print(canonical_json(summary))
        return 0

    if args.command == "bench":
        summary = bench_search(
            args.url,
            concurrency=args.concurrency,
            requests=args.requests,
            top_k=args.top_k,
            query=args.query,
            timeout_s=args.timeout,
            max_retries=args.retries,
        )
        print(canonical_json(summary))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "BACKEND_ALIASES",
    "DEFAULT_BACKOFF_S",
    "DEFAULT_MAX_BACKOFF_S",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DEFAULT_SNIPPET_CHARS",
    "DEFAULT_TASK_DEADLINE_S",
    "DEFAULT_TOP_K",
    "E5FaissBackend",
    "FASTAPI_AVAILABLE",
    "InMemorySearchBackend",
    "SEARCH_R1_MAX_ROUNDS",
    "SearchBackendError",
    "SearchClient",
    "SearchEnv",
    "app",
    "backend_from_env",
    "bench_search",
    "build_e5_index",
    "create_app",
    "load_documents",
    "main",
    "make_snippet",
    "normalise_search_config",
    "serve",
]
