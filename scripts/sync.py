#!/usr/bin/env python3
"""
Incremental sync: for each registered namespace, check whether new commits have
landed since the last index and re-embed only the changed files.

Falls back to a full reindex when the diff exceeds GitHub's compare API limits
(> 250 commits or >= 300 files changed).

Usage:
    python scripts/sync.py                     # sync all registered namespaces
    python scripts/sync.py --namespace tupaia  # sync one namespace

Environment variables:
    DATABASE_URL   — PostgreSQL connection string
    VOYAGE_API_KEY — Voyage AI API key
    GITHUB_TOKEN   — GitHub API token (optional; raises rate limit from 60 to 5000 req/hr)
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rag.db import (
    delete_file_chunks,
    get_conn,
    get_namespaces,
    register_namespace,
    setup_meta_table,
    update_namespace_sha,
    upsert_chunks,
)
from rag.github import get_changed_files, get_latest_sha, parse_repo_url
from rag.query import embed_batch
from scripts.ingest import EMBED_BATCH, chunk_file, ingest, should_skip


def _full_reindex(repo_url: str, namespace: str) -> None:
    """Clone and fully reindex. Caller updates the commit SHA afterwards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        print(f"  Cloning {repo_url} (shallow)...")
        subprocess.run(["git", "clone", "--depth=1", repo_url, repo_path], check=True)
        ingest(repo_path, namespace)  # repo_url omitted — caller handles SHA registration


def sync_namespace(conn, ns: dict, token: str | None) -> None:
    namespace = ns["namespace"]
    repo_url = ns["repo_url"]
    last_sha = ns["last_commit_sha"]

    owner, repo = parse_repo_url(repo_url)
    latest_sha = get_latest_sha(owner, repo, token)

    if latest_sha == last_sha:
        print(f"[{namespace}] Up to date ({latest_sha[:8]})")
        return

    # No previous SHA recorded — do a full reindex to establish a baseline
    if last_sha is None:
        print(f"[{namespace}] No baseline commit — running full reindex")
        _full_reindex(repo_url, namespace)
        register_namespace(conn, namespace, repo_url, latest_sha)
        return

    print(f"[{namespace}] New commits {last_sha[:8]}...{latest_sha[:8]}, checking diff...")
    changed, deleted, too_large = get_changed_files(owner, repo, last_sha, latest_sha, token)

    if too_large:
        print(f"[{namespace}] Diff too large — falling back to full reindex")
        _full_reindex(repo_url, namespace)
        update_namespace_sha(conn, namespace, latest_sha)
        return

    print(f"[{namespace}] {len(changed)} changed, {len(deleted)} deleted files")

    if not changed and not deleted:
        update_namespace_sha(conn, namespace, latest_sha)
        print(f"[{namespace}] Up to date (no indexed files changed)")
        return

    code_table = f"{namespace}_code"
    docs_table = f"{namespace}_docs"
    tables = [code_table, docs_table]

    # Remove stale chunks for every changed or deleted file
    total_deleted = 0
    for rel_path in changed + deleted:
        n = delete_file_chunks(conn, tables, rel_path)
        if n:
            print(f"  Deleted {n} stale chunks: {rel_path}")
        total_deleted += n
    print(f"  {total_deleted} stale chunks removed")

    # Re-embed changed files — requires a checkout for file content
    all_chunks: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        print(f"  Cloning {repo_url} (shallow)...")
        subprocess.run(["git", "clone", "--depth=1", repo_url, repo_path], check=True)
        repo_root = Path(repo_path)

        for rel_path in changed:
            file_path = repo_root / rel_path
            if not file_path.exists() or should_skip(Path(rel_path)):
                continue
            chunks = chunk_file(file_path, repo_root, namespace)
            all_chunks.extend(chunks)

    if all_chunks:
        code_chunks = [c for c in all_chunks if c["table"] == code_table]
        doc_chunks = [c for c in all_chunks if c["table"] == docs_table]
        print(f"  New chunks: {len(all_chunks)} ({len(code_chunks)} code, {len(doc_chunks)} docs)")

        for table, chunks in [(code_table, code_chunks), (docs_table, doc_chunks)]:
            if not chunks:
                continue
            print(f"  Upserting {len(chunks)} chunks to {table}...")
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i:i + EMBED_BATCH]
                embeddings = embed_batch([c["text"] for c in batch], input_type="document")
                upsert_chunks(conn, table, batch, embeddings)

    update_namespace_sha(conn, namespace, latest_sha)
    print(f"[{namespace}] Sync complete → {latest_sha[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental sync of registered namespaces")
    parser.add_argument(
        "--namespace", default=None,
        help="Sync only this namespace (default: all registered namespaces)",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    conn = get_conn(os.environ["DATABASE_URL"])
    setup_meta_table(conn)

    namespaces = get_namespaces(conn)

    if args.namespace:
        namespaces = [ns for ns in namespaces if ns["namespace"] == args.namespace]
        if not namespaces:
            print(f"Namespace '{args.namespace}' is not registered. Run ingest.py --repo first.")
            sys.exit(1)

    if not namespaces:
        print("No registered namespaces found. Run ingest.py --repo first.")
        sys.exit(0)

    print(f"Syncing {len(namespaces)} namespace(s)...")
    errors: list[str] = []
    for ns in namespaces:
        try:
            sync_namespace(conn, ns, token)
        except Exception as e:
            print(f"[{ns['namespace']}] ERROR: {e}")
            errors.append(ns["namespace"])

    conn.close()

    if errors:
        print(f"\nFailed: {', '.join(errors)}")
        sys.exit(1)
    print("\nAll syncs complete.")


if __name__ == "__main__":
    main()
