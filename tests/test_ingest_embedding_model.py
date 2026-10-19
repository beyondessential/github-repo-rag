"""scripts/ingest.py and scripts/sync.py selecting and enforcing a namespace's embedding model.

DB and git are faked; the Qwen test drives the real OpenAI-compatible HTTP client with a
mocked urllib, so it exercises the actual registry and request/response shape.
"""

import json
import sys
from pathlib import Path

import pytest

from scripts import ingest, sync


class FakeConn:
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "f.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


# ── Refusing to mix models ──────────────────────────────────────────────────────

def test_ac_022_refuses_a_different_model_for_an_already_indexed_namespace(monkeypatch, repo):
    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_namespace_model", lambda conn, ns: "voyage-code-3")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    with pytest.raises(RuntimeError, match="indexed with 'voyage-code-3'.*qwen3-embedding-0.6b"):
        ingest.ingest(str(repo), "ns", embedding_model="qwen3-embedding-0.6b")


def test_ac_022_refuses_a_different_model_for_an_unregistered_but_populated_namespace(monkeypatch, repo):
    """Ingested with no --repo, so never registered — its existing chunks are still assumed
    voyage-code-3, the only model that has ever existed before it."""
    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_namespace_model", lambda conn, ns: None)
    monkeypatch.setattr(ingest, "has_existing_chunks", lambda conn, tables: True)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    with pytest.raises(RuntimeError, match="indexed with 'voyage-code-3'.*qwen3-embedding-0.6b"):
        ingest.ingest(str(repo), "ns", embedding_model="qwen3-embedding-0.6b")


def test_ac_022_proceeds_for_a_genuinely_new_namespace_with_no_chunks_and_no_registry_row(monkeypatch, repo):
    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_namespace_model", lambda conn, ns: None)
    monkeypatch.setattr(ingest, "has_existing_chunks", lambda conn, tables: False)
    monkeypatch.setattr(ingest, "setup_db", lambda conn, tables, dims=1024: None)
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})

    def no_cache_read(conn, hashes):
        raise AssertionError("qwen3-embedding-0.6b has no Batch API — must not read rag_embedding_cache")

    monkeypatch.setattr(ingest, "get_cached_embeddings", no_cache_read)
    monkeypatch.setattr(ingest, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "delete_chunks_except", lambda conn, table, keep_ids: 0)
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([[1.0] * 1024]))
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    assert ingest.ingest(str(repo), "ns", embedding_model="qwen3-embedding-0.6b") is True


def test_ac_022_allows_re_ingesting_under_the_same_model(monkeypatch, repo):
    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_namespace_model", lambda conn, ns: "voyage-code-3")
    monkeypatch.setattr(ingest, "setup_db", lambda conn, tables, dims=1024: None)
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {})
    monkeypatch.setattr(ingest, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "delete_chunks_except", lambda conn, table, keep_ids: 0)
    monkeypatch.setattr(ingest, "get_model", lambda name: _ZeroModel())
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    assert ingest.ingest(str(repo), "ns", embedding_model="voyage-code-3") is True


def test_ac_021_unknown_embedding_model_raises_before_touching_the_db(monkeypatch, repo):
    def no_conn(url):
        raise AssertionError("must not open a DB connection for an unknown model")

    monkeypatch.setattr(ingest, "get_conn", no_conn)

    with pytest.raises(ValueError, match="Unknown embedding model 'bogus'"):
        ingest.ingest(str(repo), "ns", embedding_model="bogus")


# ── CLI validation ────────────────────────────────────────────────────────────

def test_ac_021_cli_rejects_an_unknown_embedding_model(monkeypatch, repo, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest.py", str(repo), "--embedding-model", "bogus"])
    with pytest.raises(SystemExit):
        ingest.main()
    assert "Unknown embedding model" in capsys.readouterr().err


def test_ac_023_cli_rejects_batch_with_a_non_batch_capable_model(monkeypatch, repo, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["ingest.py", "--repo", "https://github.com/o/r", "--batch", "--embedding-model", "qwen3-embedding-0.6b"],
    )
    with pytest.raises(SystemExit):
        ingest.main()
    assert "--batch is not supported for embedding model 'qwen3-embedding-0.6b'" in capsys.readouterr().err


# ── sync threads the namespace's own model through ──────────────────────────────

def test_ac_022_sync_threads_the_namespaces_recorded_model_through(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(sync, "shallow_clone", lambda *a, **k: None)
    monkeypatch.setattr(sync, "ingest", lambda repo_path, ns, use_batch=False, embedding_model=None:
                        seen.update(embedding_model=embedding_model) or True)
    ns = {
        "namespace": "ns", "repo_url": "https://github.com/o/r", "last_commit_sha": None,
        "embedding_model": "qwen3-embedding-0.6b",
    }
    monkeypatch.setattr(sync, "get_latest_sha", lambda owner, repo, token: "a" * 40)
    monkeypatch.setattr(sync, "get_pending_batches", lambda conn, namespace: [])
    monkeypatch.setattr(sync, "register_namespace", lambda *a, **k: None)

    sync.sync_namespace(FakeConn(), ns, None)

    assert seen["embedding_model"] == "qwen3-embedding-0.6b"


# ── Real registry, HTTP mocked: actually ingest with Qwen ───────────────────────

class _ZeroModel:
    name = "voyage-code-3"
    dims = 1024
    max_batch = 128
    supports_batch_api = True

    def embed(self, texts, input_type):
        return [[0.0]] * len(texts)


def _fake_urlopen(embeddings: list[list[float]]):
    class _Resp:
        def read(self) -> bytes:
            data = [{"embedding": e, "index": i} for i, e in enumerate(embeddings)]
            return json.dumps({"data": data}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    def urlopen(req):
        body = json.loads(req.data)
        assert len(body["input"]) <= len(embeddings), "more texts than embeddings prepared for this fake"
        return _Resp()

    return urlopen


def test_ingest_with_qwen_embeds_via_the_openai_compatible_endpoint(monkeypatch, repo):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    registered: dict = {}

    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_namespace_model", lambda conn, ns: None)
    monkeypatch.setattr(ingest, "has_existing_chunks", lambda conn, tables: False)
    monkeypatch.setattr(ingest, "setup_db", lambda conn, tables, dims=1024: registered.update(dims=dims))
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})

    def no_cache_read(conn, hashes):
        raise AssertionError("qwen3-embedding-0.6b has no Batch API — must not read rag_embedding_cache")

    monkeypatch.setattr(ingest, "get_cached_embeddings", no_cache_read)
    upserts = []
    monkeypatch.setattr(ingest, "upsert_chunks", lambda conn, table, chunks, embs: upserts.append((table, embs)))
    monkeypatch.setattr(ingest, "delete_chunks_except", lambda conn, table, keep_ids: 0)
    monkeypatch.setattr(
        ingest, "register_namespace",
        lambda conn, ns, url, sha, embedding_model=None: registered.update(embedding_model=embedding_model),
    )
    vector = [1.0] * 1024
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([vector]))
    monkeypatch.setattr(ingest.subprocess, "check_output", lambda cmd, cwd: b"deadbeef")

    result = ingest.ingest(str(repo), "ns", repo_url="https://github.com/o/r", embedding_model="qwen3-embedding-0.6b")

    assert result is True
    assert registered["dims"] == 1024
    assert registered["embedding_model"] == "qwen3-embedding-0.6b"
    assert upserts == [("ns_code", [vector])]
