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


def test_ac_012_delete_chunks_except_keeps_only_given_ids(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t_prune (id TEXT PRIMARY KEY)")
        cur.execute("INSERT INTO t_prune SELECT 'id' || g FROM generate_series(1, 100) g")
    assert db.delete_chunks_except(conn, "t_prune", [f"id{i}" for i in range(1, 91)]) == 10
    # Second call in the same session: the keep_ids temp table was dropped on commit
    assert db.delete_chunks_except(conn, "t_prune", ["id1"]) == 89


def test_ac_014_delete_chunks_except_is_a_noop_on_empty_keep_ids(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t_empty_keep (id TEXT PRIMARY KEY)")
        cur.execute("INSERT INTO t_empty_keep SELECT 'id' || g FROM generate_series(1, 20) g")
    assert db.delete_chunks_except(conn, "t_empty_keep", []) == 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM t_empty_keep")
        assert cur.fetchone()[0] == 20


# ── rag.query.search SQL ──────────────────────────────────────────────────────

def _chunk_table(conn, name: str, rows: list[tuple[str, str]], with_vector: bool = False) -> None:
    """Create a temp chunk table, left uncommitted in the caller's transaction."""
    vector_col = ", embedding vector(3)" if with_vector else ""
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE {name} (id TEXT, text TEXT, file_path TEXT, package TEXT{vector_col})")
        for i, (text, path) in enumerate(rows):
            cur.execute(
                f"INSERT INTO {name} (id, text, file_path, package) VALUES (%s, %s, %s, '')", (str(i), text, path)
            )


def test_search_fts_ranks_by_text_match_and_skips_a_missing_table(conn):
    from rag.query import search

    _chunk_table(conn, "t_search_docs", [
        ("embedding cache cache cache", "busy.md"),
        ("unrelated words", "other.md"),
        ("the embedding cache", "light.md"),
    ])
    rows = search(conn, "embedding cache", ["t_missing_code", "t_search_docs"], 5, mode="fts")
    assert [r[1] for r in rows] == ["busy.md", "light.md"]
    # The caller's uncommitted work survives the missing table
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM t_search_docs")
        assert cur.fetchone()[0] == 3


def test_search_vector_and_hybrid_sql_run_against_pgvector(conn):
    from pgvector import Vector
    from pgvector.psycopg2 import register_vector

    from rag.query import search

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        if cur.fetchone() is None:
            pytest.skip("pgvector not installed")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    _chunk_table(conn, "t_search_vec", [("alpha text", "a.py"), ("beta text", "b.py")], with_vector=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE t_search_vec SET embedding = (CASE file_path WHEN 'a.py' THEN '[1,0,0]' ELSE '[0,1,0]' END)::vector"
        )

    near_b = Vector([0.1, 1.0, 0.0])
    assert [r[1] for r in search(conn, "text", ["t_search_vec"], 2, near_b, mode="vector")] == ["b.py", "a.py"]
    assert [r[1] for r in search(conn, "beta", ["t_search_vec"], 1, near_b, mode="hybrid")] == ["b.py"]
