"""SQL checks against a real Postgres. Set TEST_DATABASE_URL to run them (pgvector not required).

Each test works in its own temporary tables, so a scratch database is enough.
"""

import os

import pytest

from rag import db

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def conn():
    import psycopg2

    c = psycopg2.connect(DATABASE_URL)
    yield c
    c.rollback()
    c.close()


def test_ac_010_sql_backfill_hash_matches_python(conn):
    texts = ["def foo():\n    return 1\n", "Māori: tēnā koe 🌏\r\n\ttabs", "x" * 5000]
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t_hash (text TEXT, content_hash TEXT)")
        for text in texts:
            cur.execute("INSERT INTO t_hash (text) VALUES (%s)", (text,))
        cur.execute(db.BACKFILL_HASH_SQL.format(table="t_hash"))
        cur.execute("SELECT text, content_hash FROM t_hash")
        rows = cur.fetchall()
    assert len(rows) == len(texts)
    assert all(h == db.content_hash(text) for text, h in rows)


def test_delete_chunks_except_keeps_only_given_ids(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t_prune (id TEXT PRIMARY KEY)")
        cur.execute("INSERT INTO t_prune SELECT 'id' || g FROM generate_series(1, 100) g")
    assert db.delete_chunks_except(conn, "t_prune", [f"id{i}" for i in range(1, 91)]) == 10
    # Second call in the same session: the keep_ids temp table was dropped on commit
    assert db.delete_chunks_except(conn, "t_prune", ["id1"]) == 89
