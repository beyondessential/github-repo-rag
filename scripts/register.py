#!/usr/bin/env python3
"""
Register (or re-register) a namespace in rag_namespaces without re-ingesting.

Useful when you have an existing index and want to tell sync.py which commit
it was built from, or to reset the baseline after a manual ingest.

Usage:
    # Register at the current HEAD of the repo:
    python scripts/register.py --repo https://github.com/beyondessential/tupaia --namespace tupaia

    # Register at a specific commit SHA (e.g. the one you actually indexed):
    python scripts/register.py --repo https://github.com/beyondessential/tupaia --namespace tupaia --sha abc1234

    # Clear the SHA so the next sync.py run triggers a full reindex:
    python scripts/register.py --repo https://github.com/beyondessential/tupaia --namespace tupaia --sha ""

Environment variables:
    DATABASE_URL  — PostgreSQL connection string
    GITHUB_TOKEN  — GitHub API token (optional; only needed when resolving HEAD)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rag.db import get_conn, register_namespace, setup_meta_table
from rag.github import get_latest_sha, parse_repo_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a namespace in rag_namespaces without re-ingesting"
    )
    parser.add_argument("--repo", required=True, help="GitHub repo URL")
    parser.add_argument("--namespace", required=True, help="Namespace to register")
    parser.add_argument(
        "--sha",
        default=None,
        help="Commit SHA to record (default: fetch current HEAD from GitHub). "
             "Pass empty string to clear and force a full reindex on next sync.",
    )
    args = parser.parse_args()

    # Resolve SHA
    if args.sha is None:
        token = os.environ.get("GITHUB_TOKEN")
        owner, repo = parse_repo_url(args.repo)
        print(f"Fetching latest SHA for {owner}/{repo}...")
        sha: str | None = get_latest_sha(owner, repo, token)
        print(f"  HEAD → {sha[:8]}")
    elif args.sha == "":
        sha = None
        print("Clearing SHA (next sync will trigger a full reindex)")
    else:
        sha = args.sha
        print(f"Using provided SHA: {sha[:8] if sha else 'None'}")

    conn = get_conn(os.environ["DATABASE_URL"])
    setup_meta_table(conn)
    register_namespace(conn, args.namespace, args.repo, sha)
    conn.close()

    status = sha[:8] if sha else "None (will full-reindex on next sync)"
    print(f"Registered '{args.namespace}' → {args.repo} @ {status}")


if __name__ == "__main__":
    main()
