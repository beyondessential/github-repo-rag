"""
Query pipeline: embed → hybrid retrieve from pgvector.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import voyageai

if TYPE_CHECKING:
    import psycopg2.extensions
    from pgvector import Vector

_DEFAULT_TABLES = ["tupaia_code", "tupaia_docs"]

SearchMode = Literal["hybrid", "vector", "fts"]
SEARCH_MODES: tuple[SearchMode, ...] = ("hybrid", "vector", "fts")

# Vector + FTS candidates fused with reciprocal rank fusion (k = 60)
_HYBRID_SQL = """
    WITH vector_results AS (
        SELECT id, text, file_path, package,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(embedding)s) AS rn
        FROM {table}
        ORDER BY embedding <=> %(embedding)s
        LIMIT 20
    ),
    fts_results AS (
        SELECT id, text, file_path, package,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(
                       to_tsvector('english', text),
                       websearch_to_tsquery('english', %(question)s)
                   ) DESC
               ) AS rn
        FROM {table}
        WHERE to_tsvector('english', text) @@
              websearch_to_tsquery('english', %(question)s)
        ORDER BY rn
        LIMIT 20
    ),
    combined AS (
        SELECT
            COALESCE(v.text, f.text) AS text,
            COALESCE(v.file_path, f.file_path) AS file_path,
            COALESCE(v.package, f.package) AS package,
            (1.0 / (60 + COALESCE(v.rn, 100)))
            + (1.0 / (60 + COALESCE(f.rn, 100))) AS score
        FROM vector_results v
        FULL OUTER JOIN fts_results f ON v.id = f.id
    )
    SELECT text, file_path, package
    FROM combined
    ORDER BY score DESC
    LIMIT %(top_k)s
"""

_VECTOR_SQL = """
    SELECT text, file_path, package
    FROM {table}
    ORDER BY embedding <=> %(embedding)s
    LIMIT %(top_k)s
"""

_FTS_SQL = """
    SELECT text, file_path, package
    FROM {table}
    WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', %(question)s)
    ORDER BY ts_rank(to_tsvector('english', text), websearch_to_tsquery('english', %(question)s)) DESC
    LIMIT %(top_k)s
"""

_SQL_BY_MODE = {"hybrid": _HYBRID_SQL, "vector": _VECTOR_SQL, "fts": _FTS_SQL}


def embed(text: str, input_type: str = "query") -> list[float]:
    """Embed a single text using voyage-code-3."""
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = client.embed([text], model="voyage-code-3", input_type=input_type)
    return result.embeddings[0]


def embed_batch(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a batch of texts (max 128 per call for voyage-code-3)."""
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = client.embed(texts, model="voyage-code-3", input_type=input_type)
    return result.embeddings


def existing_tables(conn: psycopg2.extensions.connection, tables: list[str]) -> list[str]:
    """The given tables that exist, in the order given."""
    with conn.cursor() as cur:
        cur.execute("SELECT t FROM unnest(%s::text[]) t WHERE to_regclass(t) IS NOT NULL", (tables,))
        found = {r[0] for r in cur.fetchall()}
    return [t for t in tables if t in found]


def search(
    conn: psycopg2.extensions.connection,
    question: str,
    tables: list[str],
    top_k: int,
    query_embedding: Vector | None = None,
    mode: SearchMode = "hybrid",
) -> list[tuple[str, str, str]]:
    """Return (text, file_path, package) rows: up to top_k per table, tables in the order given.

    query_embedding (a pgvector Vector) is required for the hybrid and vector modes.
    A table that does not exist yet (a namespace with no docs, say) contributes no rows.
    Ties aside, each table's rows for a smaller top_k are a prefix of its rows for a larger one.
    """
    if mode != "fts" and query_embedding is None:
        raise ValueError(f"mode {mode!r} needs a query_embedding")

    params = {"embedding": query_embedding, "question": question, "top_k": top_k}
    rows: list[tuple[str, str, str]] = []
    for table in existing_tables(conn, tables):
        with conn.cursor() as cur:
            cur.execute(_SQL_BY_MODE[mode].format(table=table), params)
            rows.extend(cur.fetchall())
    return rows


def retrieve(
    question: str,
    tables: list[str] | None = None,
    top_k: int = 6,
) -> str:
    """
    Hybrid search (vector + FTS with RRF) across the given tables.
    Returns a formatted context string, or a setup hint if empty.

    tables: DB table names to search. Defaults to ['tupaia_code', 'tupaia_docs'].
    """
    import psycopg2
    from pgvector import Vector
    from pgvector.psycopg2 import register_vector

    if tables is None:
        tables = _DEFAULT_TABLES

    query_embedding = Vector(embed(question, input_type="query"))
    conn = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=10)
    try:
        register_vector(conn)
        rows = search(conn, question, tables, top_k, query_embedding)
    finally:
        conn.close()

    if not rows:
        return "(No relevant context found — run `python scripts/ingest.py` to index the repository.)"

    chunks = []
    for text, file_path, package in rows:
        label = f"{package}/{file_path}" if package else file_path
        chunks.append(f"# {label}\n{text}")
    return "\n\n---\n\n".join(chunks)
