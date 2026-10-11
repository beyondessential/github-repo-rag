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


def get_existing_hashes(conn, tables: list[str], hashes: list[str]) -> set[str]:
    """Return the hashes that already have an embedding in any of the tables or the batch cache."""
    found: set[str] = set()
    with conn.cursor() as cur:
        for table in [*tables, "rag_embedding_cache"]:
            cur.execute(
                f"SELECT DISTINCT content_hash FROM {table} WHERE content_hash = ANY(%s)",
                (list(hashes),),
            )
            found.update(r[0] for r in cur.fetchall())
    return found


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
        # Voyage Batch API state — see specs/pipelines/batch-embedding.md
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_pending_batches (
                batch_id    TEXT PRIMARY KEY,
                namespace   TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_embedding_cache (
                content_hash  TEXT PRIMARY KEY,
                embedding     vector(1024) NOT NULL,
                batch_id      TEXT NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
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


# ── Batch embedding state ─────────────────────────────────────────────────────

def add_pending_batch(conn, batch_id: str, namespace: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rag_pending_batches (batch_id, namespace) VALUES (%s, %s)",
            (batch_id, namespace),
        )
    conn.commit()


def get_pending_batches(conn, namespace: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id FROM rag_pending_batches WHERE namespace = %s ORDER BY created_at",
            (namespace,),
        )
        return [r[0] for r in cur.fetchall()]


def delete_pending_batches(conn, batch_ids: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rag_pending_batches WHERE batch_id = ANY(%s)", (batch_ids,))
    conn.commit()


def cache_embeddings(conn, batch_id: str, rows: list[tuple[str, list[float]]]) -> None:
    """Store (content_hash, embedding) pairs from a batch output."""
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO rag_embedding_cache (content_hash, embedding, batch_id)
            VALUES %s
            ON CONFLICT (content_hash) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                batch_id = EXCLUDED.batch_id,
                created_at = NOW()
            """,
            [(h, emb, batch_id) for h, emb in rows],
            template="(%s, %s::vector, %s)",
        )
    conn.commit()


def get_cached_embeddings(conn, hashes: list[str]) -> dict[str, list[float]]:
    if not hashes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash, embedding FROM rag_embedding_cache WHERE content_hash = ANY(%s)",
            (list(hashes),),
        )
        return {h: emb for h, emb in cur.fetchall()}


def delete_cached_embeddings(conn, batch_ids: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rag_embedding_cache WHERE batch_id = ANY(%s)", (batch_ids,))
    conn.commit()


def purge_stale_cache(conn, max_age_days: int = 7) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM rag_embedding_cache WHERE created_at < NOW() - make_interval(days => %s)",
            (max_age_days,),
        )
        n = cur.rowcount
    conn.commit()
    return n
