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

from rag.db import get_conn, resolve_model, sanitise_namespace, setup_db, setup_meta_table
from scripts.ingest import replace_file_chunks


def reindex(repo_path: str, namespace: str) -> None:
    namespace = sanitise_namespace(namespace)
    conn = get_conn(os.environ["DATABASE_URL"])
    try:
        # embed_chunks reads rag_embedding_cache; resolve_model reads rag_namespaces
        setup_meta_table(conn)
        model = resolve_model(conn, namespace)
        setup_db(conn, [f"{namespace}_code", f"{namespace}_docs"], dims=model.dims)

        changed = [f for f in os.environ.get("CHANGED_FILES", "").split() if f]
        deleted = [f for f in os.environ.get("DELETED_FILES", "").split() if f]
        print(f"Changed: {len(changed)} files, Deleted: {len(deleted)} files")

        replace_file_chunks(conn, namespace, Path(repo_path).resolve(), changed, deleted, model)
    finally:
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
