"""
Query pipeline: embed → hybrid retrieve from pgvector.
"""

import os

import voyageai

_DEFAULT_TABLES = ["tupaia_code", "tupaia_docs"]


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

    chunks: list[str] = []
    for table in tables:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH vector_results AS (
                        SELECT id, text, file_path, package,
                               ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rn
                        FROM {table}
                        ORDER BY embedding <=> %s
                        LIMIT 20
                    ),
                    fts_results AS (
                        SELECT id, text, file_path, package,
                               ROW_NUMBER() OVER (
                                   ORDER BY ts_rank(
                                       to_tsvector('english', text),
                                       websearch_to_tsquery('english', %s)
                                   ) DESC
                               ) AS rn
                        FROM {table}
                        WHERE to_tsvector('english', text) @@
                              websearch_to_tsquery('english', %s)
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
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, question, question, top_k),
                )
                for text, file_path, package in cur.fetchall():
                    label = f"{package}/{file_path}" if package else file_path
                    chunks.append(f"# {label}\n{text}")
        except Exception:
            pass  # table may not exist yet

    conn.close()

    if not chunks:
        return "(No relevant context found — run `python scripts/ingest.py` to index the repository.)"

    return "\n\n---\n\n".join(chunks)


