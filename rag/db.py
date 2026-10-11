"""Database helpers: setup, upsert."""

import hashlib
import re

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values


def sanitise_namespace(namespace: str) -> str:
    """Convert a namespace to a valid PostgreSQL identifier segment.

    Replaces any character that isn't a lowercase letter, digit, or underscore
    with an underscore (e.g. 'data-lake' → 'data_lake').
    """
    return re.sub(r"[^a-z0-9_]", "_", namespace.lower())


def get_conn(database_url: str):
    """Open a psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    return conn


def setup_table(conn, table: str) -> None:
    """Create a single RAG table with HNSW, FTS, and file_path indexes."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                embedding vector(1024) NOT NULL,
                text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                package TEXT NOT NULL DEFAULT '',
                chunk_index INT NOT NULL DEFAULT 0
            )
        """)
        cur.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS chunk_index INT NOT NULL DEFAULT 0
        """)
        # content_hash lets re-indexing reuse an existing embedding for unchanged text.
        # Backfilled in SQL; must match content_hash() below.
        cur.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS content_hash TEXT
        """)
        cur.execute(f"""
            UPDATE {table}
            SET content_hash = encode(sha256(convert_to(text, 'UTF8')), 'hex')
            WHERE content_hash IS NULL
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx
            ON {table} USING hnsw (embedding vector_cosine_ops)
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_fts_idx
            ON {table} USING gin (to_tsvector('english', text))
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_file_path_idx
            ON {table} (file_path)
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_content_hash_idx
            ON {table} (content_hash)
        """)
    conn.commit()


def setup_db(conn, tables: list[str]) -> None:
    """Create all RAG tables and indexes."""
    for table in tables:
        setup_table(conn, table)


def content_hash(text: str) -> str:
    """SHA-256 hex digest of a chunk's text, matching the SQL backfill in setup_table()."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embeddings_by_hash(conn, table: str, hashes: list[str]) -> dict[str, list[float]]:
    """Return {content_hash: embedding} for any hashes already stored in the table."""
    if not hashes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (content_hash) content_hash, embedding
            FROM {table}
            WHERE content_hash = ANY(%s)
            """,
            (list(hashes),),
        )
        return {h: emb for h, emb in cur.fetchall()}


def upsert_chunks(
    conn, table: str, chunks: list[dict], embeddings: list[list[float]], commit: bool = True
) -> None:
    """Upsert a batch of chunks into a RAG table."""
    rows = [
        (
            chunk["id"], emb, chunk["text"], chunk["file_path"], chunk["package"],
            chunk.get("chunk_index", 0), content_hash(chunk["text"]),
        )
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {table} (id, embedding, text, file_path, package, chunk_index, content_hash)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                text = EXCLUDED.text,
                file_path = EXCLUDED.file_path,
                package = EXCLUDED.package,
                chunk_index = EXCLUDED.chunk_index,
                content_hash = EXCLUDED.content_hash
            """,
            rows,
            template="(%s, %s::vector, %s, %s, %s, %s, %s)",
        )
    if commit:
        conn.commit()


def delete_file_chunks(conn, tables: list[str], file_path: str, commit: bool = True) -> int:
    """Delete all chunks for a file across the given tables. Returns count deleted."""
    total = 0
    for table in tables:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE file_path = %s", (file_path,))
            total += cur.rowcount
    if commit:
        conn.commit()
    return total


# ── Namespace registry ────────────────────────────────────────────────────────

def setup_meta_table(conn) -> None:
    """Create the namespace registry table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_namespaces (
                namespace        TEXT PRIMARY KEY,
                repo_url         TEXT NOT NULL,
                last_commit_sha  TEXT,
                last_indexed_at  TIMESTAMPTZ
            )
        """)
    conn.commit()


def register_namespace(conn, namespace: str, repo_url: str, commit_sha: str | None) -> None:
    """Upsert a namespace entry with its latest indexed commit SHA."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_namespaces (namespace, repo_url, last_commit_sha, last_indexed_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (namespace) DO UPDATE SET
                repo_url        = EXCLUDED.repo_url,
                last_commit_sha = EXCLUDED.last_commit_sha,
                last_indexed_at = NOW()
            """,
            (namespace, repo_url, commit_sha),
        )
    conn.commit()


def get_namespaces(conn) -> list[dict]:
    """Return all registered namespaces as a list of dicts."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT namespace, repo_url, last_commit_sha FROM rag_namespaces ORDER BY namespace"
        )
        rows = cur.fetchall()
    return [{"namespace": r[0], "repo_url": r[1], "last_commit_sha": r[2]} for r in rows]


def update_namespace_sha(conn, namespace: str, commit_sha: str) -> None:
    """Update the last indexed commit SHA for an existing namespace."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE rag_namespaces
            SET last_commit_sha = %s, last_indexed_at = NOW()
            WHERE namespace = %s
            """,
            (commit_sha, namespace),
        )
    conn.commit()
