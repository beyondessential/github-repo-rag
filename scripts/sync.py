#!/usr/bin/env python3
"""
Incremental sync: for each registered namespace, check whether new commits have
landed since the last index and re-embed only the changed files.

Falls back to a full reindex when the diff exceeds GitHub's compare API limits
(> 250 commits or >= 300 files changed). A large full reindex is submitted to the
Voyage Batch API and finished by a later run once the batch completes
(specs/pipelines/embedding-sync.md).

Usage:
    python scripts/sync.py                     # sync all registered namespaces
    python scripts/sync.py --namespace tupaia  # sync one namespace

Environment variables:
    DATABASE_URL   — PostgreSQL connection string
    VOYAGE_API_KEY — Voyage AI API key
    VOYAGE_BATCH_THRESHOLD — new texts before a full reindex uses the Voyage Batch API (default 2000)
    GITHUB_TOKEN   — GitHub API token (optional; raises rate limit from 60 to 5000 req/hr)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rag import voyage_batch
from rag.db import (
    cache_embeddings,
    delete_cached_embeddings,
    delete_pending_batches,
    get_conn,
    get_namespaces,
    get_pending_batches,
    purge_stale_cache,
    register_namespace,
    setup_db,
    setup_meta_table,
    update_namespace_sha,
)
from rag.github import get_changed_files, get_latest_sha, parse_repo_url, shallow_clone
from scripts.ingest import ingest, replace_file_chunks

CACHE_PAGE = 1000  # batch output rows per cache insert


def _full_reindex(repo_url: str, namespace: str, token: str | None, use_batch: bool = True) -> bool:
    """Clone and fully reindex. Returns False if a Voyage batch was submitted instead.

    Caller updates the commit SHA afterwards, only when this returns True.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        print(f"  Cloning {repo_url} (shallow)...")
        shallow_clone(repo_url, repo_path, token)
        # repo_url omitted — caller handles SHA registration
        return ingest(repo_path, namespace, use_batch=use_batch)


def _collect_batches(conn, namespace: str, batch_ids: list[str]) -> bool:
    """Load finished batch output into the embedding cache. Returns False while any batch is in flight."""
    batches = [voyage_batch.get_batch(b) for b in batch_ids]

    # BL-006: wait for every batch in the namespace to finish
    in_flight = [b for b in batches if b["status"] in voyage_batch.IN_FLIGHT]
    if in_flight:
        for b in in_flight:
            counts = b.get("request_counts") or {}
            progress = f"{counts.get('completed', 0)}/{counts.get('total', '?')}"
            print(f"[{namespace}] Batch {b['id']} {b['status']} ({progress})")
        return False

    failed = []
    for b in batches:
        if b.get("output_file_id"):
            # Cache in pages so a 100K-line output never sits in memory at once
            cached = 0
            page: list[tuple[str, list[float]]] = []
            for row in voyage_batch.iter_output_embeddings(b["output_file_id"]):
                page.append(row)
                if len(page) >= CACHE_PAGE:
                    cache_embeddings(conn, b["id"], page)
                    cached += len(page)
                    page = []
            if page:
                cache_embeddings(conn, b["id"], page)
                cached += len(page)
            print(f"[{namespace}] Batch {b['id']} {b['status']}: cached {cached} embeddings")
        if b["status"] in voyage_batch.FAILED:
            failed.append(b)

    # BL-009: drop every pending row (completed output stays cached) so the next run's full
    # reindex resubmits only the uncached texts instead of embedding them in real time
    if failed:
        delete_pending_batches(conn, batch_ids)
        detail = ", ".join(f"{b['id']} ({b['status']}: {b.get('errors')})" for b in failed)
        raise RuntimeError(f"Voyage batch failed: {detail}")
    return True


def sync_namespace(conn, ns: dict, token: str | None) -> None:
    namespace = ns["namespace"]
    repo_url = ns["repo_url"]
    last_sha = ns["last_commit_sha"]

    owner, repo = parse_repo_url(repo_url)
    latest_sha = get_latest_sha(owner, repo, token)

    batch_ids = get_pending_batches(conn, namespace)
    if batch_ids:
        if not _collect_batches(conn, namespace, batch_ids):
            return
        # BL-007: real-time full reindex, reading batch results from the cache
        print(f"[{namespace}] Batches complete — running full reindex from cache")
        _full_reindex(repo_url, namespace, token, use_batch=False)
        # BL-008: advance the SHA, then clear batch state
        register_namespace(conn, namespace, repo_url, latest_sha)
        delete_pending_batches(conn, batch_ids)
        delete_cached_embeddings(conn, batch_ids)
        print(f"[{namespace}] Sync complete → {latest_sha[:8]}")
        return

    if latest_sha == last_sha:
        print(f"[{namespace}] Up to date ({latest_sha[:8]})")
        return

    # No previous SHA recorded — do a full reindex to establish a baseline
    if last_sha is None:
        print(f"[{namespace}] No baseline commit — running full reindex")
        if _full_reindex(repo_url, namespace, token):
            register_namespace(conn, namespace, repo_url, latest_sha)
        return

    print(f"[{namespace}] New commits {last_sha[:8]}...{latest_sha[:8]}, checking diff...")
    changed, deleted, too_large = get_changed_files(owner, repo, last_sha, latest_sha, token)

    if too_large:
        print(f"[{namespace}] Diff too large — falling back to full reindex")
        if _full_reindex(repo_url, namespace, token):
            update_namespace_sha(conn, namespace, latest_sha)
        return

    print(f"[{namespace}] {len(changed)} changed, {len(deleted)} deleted files")

    if not changed and not deleted:
        update_namespace_sha(conn, namespace, latest_sha)
        print(f"[{namespace}] Up to date (no indexed files changed)")
        return

    # Adds content_hash to tables created before embedding reuse existed
    setup_db(conn, [f"{namespace}_code", f"{namespace}_docs"])

    # Re-embed changed files — requires a checkout for file content
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        print(f"  Cloning {repo_url} (shallow)...")
        shallow_clone(repo_url, repo_path, token)
        replace_file_chunks(conn, namespace, Path(repo_path), changed, deleted)

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
    # BL-010: drop week-old batch results
    purged = purge_stale_cache(conn)
    if purged:
        print(f"Purged {purged} stale batch embeddings")

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
            # Clear any aborted transaction so later namespaces can still use the connection
            conn.rollback()
            print(f"[{ns['namespace']}] ERROR: {e}")
            errors.append(ns["namespace"])

    conn.close()

    if errors:
        print(f"\nFailed: {', '.join(errors)}")
        sys.exit(1)
    print("\nAll syncs complete.")


if __name__ == "__main__":
    main()
