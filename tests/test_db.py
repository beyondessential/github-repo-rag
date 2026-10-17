"""Mocked unit tests for rag/db.py logic that doesn't need a real Postgres connection."""

import psycopg2.errors
import pytest

from rag import db
from rag.embeddings import DEFAULT_MODEL_NAME


class _FakeCursor:
    def __init__(self, log: list[str], content_hash_exists: bool) -> None:
        self.log = log
        self.content_hash_exists = content_hash_exists

    def execute(self, sql, params=None) -> None:
        self.log.append(" ".join(sql.split()))

    def fetchone(self):
        return (1,) if self.content_hash_exists else None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeConn:
    def __init__(self, content_hash_exists: bool) -> None:
        self.log: list[str] = []
        self._content_hash_exists = content_hash_exists

    def cursor(self):
        return _FakeCursor(self.log, self._content_hash_exists)

    def commit(self) -> None:
        pass


def test_ac_018_setup_table_skips_migration_when_content_hash_exists():
    conn = _FakeConn(content_hash_exists=True)
    db.setup_table(conn, "ns_code")
    assert not any("ALTER TABLE" in q for q in conn.log)
    assert not any(q.startswith("UPDATE ns_code") for q in conn.log)
    # The non-migration indexes still get their (idempotent) IF NOT EXISTS check
    assert any("ns_code_embedding_idx" in q for q in conn.log)


def test_ac_018_setup_table_runs_migration_when_content_hash_missing():
    conn = _FakeConn(content_hash_exists=False)
    db.setup_table(conn, "ns_code")
    assert any("ALTER TABLE" in q and "content_hash" in q for q in conn.log)
    assert any(q.startswith("UPDATE ns_code") for q in conn.log)
    assert any("ns_code_content_hash_idx" in q for q in conn.log)


# ── rag_namespaces.embedding_model migration ───────────────────────────────────
# _FakeCursor.fetchone() reports whether the checked column already exists; here
# that's rag_namespaces.embedding_model rather than setup_table()'s content_hash.

def test_setup_meta_table_skips_migration_when_embedding_model_exists():
    conn = _FakeConn(content_hash_exists=True)
    db.setup_meta_table(conn)
    assert not any("ALTER TABLE rag_namespaces" in q for q in conn.log)


def test_setup_meta_table_adds_embedding_model_with_the_default_when_missing():
    conn = _FakeConn(content_hash_exists=False)
    db.setup_meta_table(conn)
    migration = next(q for q in conn.log if "ALTER TABLE rag_namespaces" in q)
    assert "embedding_model" in migration and f"DEFAULT '{DEFAULT_MODEL_NAME}'" in migration


# ── Namespace model lookup and registration ────────────────────────────────────

class _RowCursor:
    def __init__(self, row=None, raises=None) -> None:
        self.row = row
        self.raises = raises
        self.executed: list[tuple] = []

    def execute(self, sql, params=None) -> None:
        if self.raises:
            raise self.raises
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _RowConn:
    def __init__(self, row=None, raises=None) -> None:
        self._cursor = _RowCursor(row, raises)
        self.rollbacks = 0
        self.commits = 0

    def cursor(self):
        return self._cursor

    def rollback(self) -> None:
        self.rollbacks += 1

    def commit(self) -> None:
        self.commits += 1


def test_get_namespace_model_returns_the_stored_model():
    conn = _RowConn(row=("qwen3-embedding-0.6b",))
    assert db.get_namespace_model(conn, "ns") == "qwen3-embedding-0.6b"


def test_get_namespace_model_returns_none_when_unregistered():
    conn = _RowConn(row=None)
    assert db.get_namespace_model(conn, "ns") is None


def test_get_namespace_model_returns_none_and_rolls_back_when_table_missing():
    conn = _RowConn(raises=psycopg2.errors.UndefinedTable())
    assert db.get_namespace_model(conn, "ns") is None
    assert conn.rollbacks == 1


@pytest.mark.parametrize(
    "row, expected", [(("qwen3-embedding-0.6b",), "qwen3-embedding-0.6b"), (None, DEFAULT_MODEL_NAME)]
)
def test_resolve_model_defaults_to_voyage_for_an_unregistered_namespace(row, expected):
    model = db.resolve_model(_RowConn(row=row), "ns")
    assert model.name == expected


def test_ac_024_register_namespace_without_embedding_model_leaves_it_untouched_on_conflict():
    conn = _RowConn()
    db.register_namespace(conn, "ns", "https://example.com/r", "sha1")
    sql, params = conn._cursor.executed[0]
    assert "embedding_model" not in sql
    assert params == ("ns", "https://example.com/r", "sha1")
    assert conn.commits == 1


def test_ac_024_register_namespace_with_embedding_model_sets_it_on_insert_and_conflict():
    conn = _RowConn()
    db.register_namespace(conn, "ns", "https://example.com/r", "sha1", embedding_model="qwen3-embedding-0.6b")
    sql, params = conn._cursor.executed[0]
    assert "embedding_model" in sql and "EXCLUDED.embedding_model" in sql
    assert params == ("ns", "https://example.com/r", "sha1", "qwen3-embedding-0.6b")


def test_get_namespaces_includes_embedding_model():
    class _ListCursor(_RowCursor):
        def fetchall(self):
            return [("ns", "https://example.com/r", "sha1", "qwen3-embedding-0.6b")]

    class _ListConn(_RowConn):
        def __init__(self) -> None:
            super().__init__()
            self._cursor = _ListCursor()

    assert db.get_namespaces(_ListConn()) == [
        {
            "namespace": "ns",
            "repo_url": "https://example.com/r",
            "last_commit_sha": "sha1",
            "embedding_model": "qwen3-embedding-0.6b",
        }
    ]
