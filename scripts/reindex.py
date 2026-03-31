#!/usr/bin/env python3
"""
Incremental reindex: re-embed changed files, delete stale chunks by file_path.

Called from GitHub Actions after a push to the indexed repository.

Usage:
    python scripts/reindex.py /path/to/repo --namespace tupaia

Environment variables:
    CHANGED_FILES  — space-separated list of changed file paths (relative to repo root)
    DELETED_FILES  — space-separated list of deleted file paths (relative to repo root)
    DATABASE_URL   — PostgreSQL connection string
    VOYAGE_API_KEY — Voyage AI API key
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from scripts.ingest import EMBED_BATCH, chunk_file, should_skip
from rag.db import delete_file_chunks, get_conn, upsert_chunks
from rag.query import embed_batch


def reindex(repo_path: str, namespace: str) -> None:
    code_table = f"{namespace}_code"
    docs_table = f"{namespace}_docs"
    tables = [code_table, docs_table]

    conn = get_conn(os.environ["DATABASE_URL"])
    repo_root = Path(repo_path).resolve()

    changed = [f for f in os.environ.get("CHANGED_FILES", "").split() if f]
    deleted = [f for f in os.environ.get("DELETED_FILES", "").split() if f]
    print(f"Changed: {len(changed)} files, Deleted: {len(deleted)} files")

    # Delete stale chunks for changed and deleted files
    total_deleted = 0
    for rel_path in changed + deleted:
        n = delete_file_chunks(conn, tables, rel_path)
        if n:
            print(f"  Deleted {n} chunks for {rel_path}")
        total_deleted += n
    print(f"Deleted {total_deleted} stale chunks total")

    # Re-chunk and re-embed changed files
    all_chunks: list[dict] = []
    for rel_path in changed:
        file_path = repo_root / rel_path
        if not file_path.exists():
            continue
        if should_skip(Path(rel_path)):
            continue
        chunks = chunk_file(file_path, repo_root, namespace)
        if chunks:
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No new chunks to index.")
        conn.close()
        return

    code_chunks = [c for c in all_chunks if c["table"] == code_table]
    doc_chunks = [c for c in all_chunks if c["table"] == docs_table]
    print(f"\nNew chunks: {len(all_chunks)} ({len(code_chunks)} code, {len(doc_chunks)} docs)")

    for table, chunks in [(code_table, code_chunks), (docs_table, doc_chunks)]:
        if not chunks:
            continue
        print(f"Upserting {len(chunks)} chunks to {table}...")
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            embeddings = embed_batch([c["text"] for c in batch], input_type="document")
            upsert_chunks(conn, table, batch, embeddings)

    conn.close()
    print("Reindex complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental reindex of changed files")
    parser.add_argument("repo_path", help="Path to the repo checkout")
    parser.add_argument(
        "--namespace", default="tupaia",
        help="Table prefix (default: tupaia)",
    )
    args = parser.parse_args()
    reindex(args.repo_path, args.namespace)


if __name__ == "__main__":
    main()
