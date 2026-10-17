"""
Query pipeline: embed → hybrid retrieve from pgvector.
"""

import os
from typing import Literal

import voyageai

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


def search(
    conn,
    question: str,
    tables: list[str],
    top_k: int,
    query_embedding=None,
    mode: SearchMode = "hybrid",
) -> list[tuple[str, str, str]]:
    """Return (text, file_path, package) rows: up to top_k per table, tables in the order given.

    query_embedding (a pgvector Vector) is required for the hybrid and vector modes.
    A table that does not exist yet contributes no rows.
    """
    import psycopg2.errors

    if mode != "fts" and query_embedding is None:
        raise ValueError(f"mode {mode!r} needs a query_embedding")

    params = {"embedding": query_embedding, "question": question, "top_k": top_k}
    rows: list[tuple[str, str, str]] = []
    for table in tables:
        try:
            with conn.cursor() as cur:
                cur.execute(_SQL_BY_MODE[mode].format(table=table), params)
                rows.extend(cur.fetchall())
        except psycopg2.errors.UndefinedTable:
            # A namespace's _docs or _code table may not exist yet. Roll back so
            # the aborted transaction does not fail the remaining tables.
            conn.rollback()
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
    register_vector(conn)
    try:
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
