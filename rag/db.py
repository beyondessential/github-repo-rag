"""Database helpers: setup, upsert."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

if TYPE_CHECKING:
    from rag.embeddings import EmbeddingModel


def sanitise_namespace(namespace: str) -> str:
    """Convert a namespace to a valid PostgreSQL identifier segment.

    Replaces any character that isn't a lowercase letter, digit, or underscore
    with an underscore (e.g. 'data-lake' → 'data_lake').
    """
    return re.sub(r"[^a-z0-9_]", "_", namespace.lower())


# BL-011: must produce the same digest as content_hash()
BACKFILL_HASH_SQL = """
    UPDATE {table}
    SET content_hash = encode(sha256(convert_to(text, 'UTF8')), 'hex')
    WHERE content_hash IS NULL
"""


def get_conn(database_url: str):
    """Open a psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    return conn


def setup_table(conn, table: str, dims: int = 1024) -> None:
    """Create a single RAG table with HNSW, FTS, file_path and content_hash indexes.

    dims is only used on first creation — CREATE TABLE IF NOT EXISTS is a no-op against
    an existing table, so this never changes an existing table's vector dimension. A
    namespace's embedding model is fixed at ingest (see rag.db.register_namespace), so
    this should only ever be called with the dims of the model a namespace is already
    using. Inserting a vector of the wrong dimension is rejected by Postgres itself.

    The column/backfill migration below runs only while content_hash is missing —
    this is called on every ingest and sync tick (hourly, per namespace), and an
    ALTER TABLE takes its lock before checking IF NOT EXISTS, so re-running it
    unconditionally would repeat that lock acquisition against a live table the
    MCP server concurrently reads, forever, for no effect after the first run.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                embedding vector({dims}) NOT NULL,
                text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                package TEXT NOT NULL DEFAULT '',
                chunk_index INT NOT NULL DEFAULT 0
            )
        """)
        # BL-019: only migrate while content_hash is missing
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'content_hash'",
            (table,),
        )
        if cur.fetchone() is None:
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
            cur.execute(BACKFILL_HASH_SQL.format(table=table))
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_content_hash_idx
                ON {table} (content_hash)
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
    conn.commit()


def setup_db(conn, tables: list[str], dims: int = 1024) -> None:
    """Create all RAG tables and indexes. All of a namespace's tables share one model's dims."""
    for table in tables:
        setup_table(conn, table, dims=dims)


# BL-011
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
            ORDER BY content_hash, id
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



def delete_chunks_except(conn, table: str, keep_ids: list[str]) -> int:
    """Delete every chunk in the table whose id is not in keep_ids. Returns count deleted.

    BL-016: a no-op when keep_ids is empty. A run that found zero chunks for this
    table (a genuinely empty repo subtree, or a partial/broken checkout) must never
    be read as "keep nothing" — that would silently empty an already-indexed table.
    """
    if not keep_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE keep_ids (id TEXT PRIMARY KEY) ON COMMIT DROP")
        execute_values(cur, "INSERT INTO keep_ids (id) VALUES %s ON CONFLICT DO NOTHING", [(i,) for i in keep_ids])
        cur.execute(f"DELETE FROM {table} t WHERE NOT EXISTS (SELECT 1 FROM keep_ids k WHERE k.id = t.id)")
        n = cur.rowcount
    conn.commit()
    return n

# ── Namespace registry ────────────────────────────────────────────────────────

def setup_meta_table(conn) -> None:
    """Create the namespace registry table if it does not exist.

    The embedding_model migration below runs only while the column is missing, for
    the same reason as setup_table()'s content_hash migration: this runs on every
    ingest and sync tick, and an unconditional ALTER TABLE would repeat its lock
    acquisition forever for no effect after the first run.
    """
    from rag.embeddings import DEFAULT_MODEL_NAME

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_namespaces (
                namespace        TEXT PRIMARY KEY,
                repo_url         TEXT NOT NULL,
                last_commit_sha  TEXT,
                last_indexed_at  TIMESTAMPTZ
            )
        """)
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'rag_namespaces' AND column_name = 'embedding_model'"
        )
        if cur.fetchone() is None:
            # Existing namespaces were all indexed with voyage-code-3, the only model
            # there was before this column existed.
            cur.execute(f"""
                ALTER TABLE rag_namespaces
                ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT '{DEFAULT_MODEL_NAME}'
            """)
        # Voyage Batch API state — see specs/pipelines/embedding-sync.md
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


def register_namespace(
    conn, namespace: str, repo_url: str, commit_sha: str | None, embedding_model: str | None = None
) -> None:
    """Upsert a namespace entry with its latest indexed commit SHA.

    BL-025: embedding_model is written only when given explicitly. Omitting it (the default)
    leaves an existing namespace's recorded model untouched and a new one gets the
    column default (voyage-code-3) — re-registering a SHA must never silently reset
    which model a namespace is indexed with.
    """
    with conn.cursor() as cur:
        if embedding_model is not None:
            cur.execute(
                """
                INSERT INTO rag_namespaces (namespace, repo_url, last_commit_sha, last_indexed_at, embedding_model)
                VALUES (%s, %s, %s, NOW(), %s)
                ON CONFLICT (namespace) DO UPDATE SET
                    repo_url        = EXCLUDED.repo_url,
                    last_commit_sha = EXCLUDED.last_commit_sha,
                    last_indexed_at = NOW(),
                    embedding_model = EXCLUDED.embedding_model
                """,
                (namespace, repo_url, commit_sha, embedding_model),
            )
        else:
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
            "SELECT namespace, repo_url, last_commit_sha, embedding_model "
            "FROM rag_namespaces ORDER BY namespace"
        )
        rows = cur.fetchall()
    return [
        {"namespace": r[0], "repo_url": r[1], "last_commit_sha": r[2], "embedding_model": r[3]} for r in rows
    ]


def has_existing_chunks(conn, tables: list[str]) -> bool:
    """True if any of the tables already has a row. A table that does not exist yet counts as empty.

    BL-023: used to infer a namespace's model when it predates the embedding_model column, or was
    never registered at all (ingested with no --repo, so rag_namespaces has no row for it)
    — every such namespace is voyage-code-3, the only model that has ever existed before it.

    Checks which tables exist up front rather than catching UndefinedTable, so a missing
    table never rolls back other work a caller may already have done on this connection.
    """
    from rag.query import existing_tables

    for table in existing_tables(conn, tables):
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
            if cur.fetchone() is not None:
                return True
    return False


def get_namespace_model(conn, namespace: str) -> str | None:
    """The embedding model recorded for a namespace.

    None if the namespace is not registered, or if rag_namespaces does not exist yet
    (a namespace's _code/_docs tables can exist without it in principle, though the
    current ingest path always creates both together).
    """
    import psycopg2.errors

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding_model FROM rag_namespaces WHERE namespace = %s", (namespace,))
            row = cur.fetchone()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return None
    return row[0] if row else None


def resolve_model(conn, namespace: str) -> EmbeddingModel:
    """The EmbeddingModel a namespace is indexed with, defaulting to voyage-code-3
    (DEFAULT_MODEL_NAME) for a namespace with no recorded model.
    """
    from rag.embeddings import DEFAULT_MODEL_NAME, get_model

    return get_model(get_namespace_model(conn, namespace) or DEFAULT_MODEL_NAME)


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
