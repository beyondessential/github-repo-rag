"""Mocked unit tests for rag/db.py logic that doesn't need a real Postgres connection."""

from rag import db


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
