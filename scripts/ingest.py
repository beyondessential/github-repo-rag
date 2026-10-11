#!/usr/bin/env python3
"""
Ingestion pipeline: clone a GitHub repo → chunk → embed → upsert to pgvector.

Usage:
    # Auto-clones to a temp directory (default branch):
    python scripts/ingest.py --repo https://github.com/beyondessential/tupaia

    # Clone a specific release tag or branch:
    python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref 2.50.5
    python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref main

    # Use an existing local checkout:
    python scripts/ingest.py /path/to/tupaia

    # Custom namespace (creates {namespace}_code and {namespace}_docs tables):
    python scripts/ingest.py --namespace tamanu /path/to/tamanu

Environment variables:
    DATABASE_URL   — PostgreSQL connection string (must have pgvector extension)
    VOYAGE_API_KEY — Voyage AI API key
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rag.db import (
    content_hash,
    delete_file_chunks,
    get_conn,
    get_embeddings_by_hash,
    register_namespace,
    sanitise_namespace,
    setup_db,
    setup_meta_table,
    upsert_chunks,
)
from rag.github import shallow_clone
from rag.query import embed_batch

# ── Skip rules ────────────────────────────────────────────────────────────────

SKIP_DIRS = {
    "node_modules", "dist", "build", ".git", "__pycache__",
    ".turbo", ".next", "coverage", ".nyc_output", "storybook-static",
    ".yarn", "vendor",
}
SKIP_SUFFIXES = {
    ".lock", ".snap", ".map", ".ico", ".png", ".jpg", ".jpeg",
    ".gif", ".svg", ".woff", ".woff2", ".ttf", ".eot", ".pdf",
    ".min.js", ".min.css",
}
SKIP_FILENAMES = {"yarn.lock", "package-lock.json", "pnpm-lock.yaml"}

# Source code → {namespace}_code table, ~512 token chunks
CODE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".go", ".rs", ".sql"}
CODE_CHUNK_SIZE = 2000    # chars (~512 tokens)
CODE_CHUNK_OVERLAP = 200

# Docs → {namespace}_docs table, ~1024 token chunks
DOC_SUFFIXES = {".md", ".txt", ".rst", ".yml", ".yaml", ".toml"}
DOC_CHUNK_SIZE = 4000     # chars (~1024 tokens)
DOC_CHUNK_OVERLAP = 400

EMBED_BATCH = 64  # texts per voyage API call
INDEX_GROUP = 1000  # chunks per hash lookup + upsert during a full ingest

_LANG_MAP = {
    ".ts": Language.TS, ".tsx": Language.TS,
    ".js": Language.JS, ".jsx": Language.JS,
    ".py": Language.PYTHON,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(rel: Path) -> bool:
    for part in rel.parts:
        if part in SKIP_DIRS:
            return True
    if rel.name in SKIP_FILENAMES:
        return True
    name_lower = rel.name.lower()
    for suf in SKIP_SUFFIXES:
        if name_lower.endswith(suf):
            return True
    return False


def get_package_name(file_path: Path, repo_root: Path) -> str:
    """Walk up from file_path looking for the nearest package.json with a 'name' field."""
    candidate = file_path.parent
    while candidate != repo_root and candidate != candidate.parent:
        pkg_json = candidate / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
                name = data.get("name", "")
                if name:
                    return name
            except Exception:
                pass
        candidate = candidate.parent
    return ""


def chunk_file(file_path: Path, repo_root: Path, namespace: str) -> list[dict]:
    """Read and chunk a file. Returns list of chunk dicts with id, text, metadata, table."""
    ext = file_path.suffix.lower()

    if ext in CODE_SUFFIXES:
        table = f"{namespace}_code"
        lang = _LANG_MAP.get(ext)
        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=CODE_CHUNK_SIZE,
                chunk_overlap=CODE_CHUNK_OVERLAP,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CODE_CHUNK_SIZE,
                chunk_overlap=CODE_CHUNK_OVERLAP,
            )
    elif ext in DOC_SUFFIXES:
        table = f"{namespace}_docs"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DOC_CHUNK_SIZE,
            chunk_overlap=DOC_CHUNK_OVERLAP,
        )
    else:
        return []

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if not text.strip():
        return []

    rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    package = get_package_name(file_path, repo_root)
    chunks_text = splitter.split_text(text)

    return [
        {
            "id": hashlib.sha256(f"{namespace}:{rel_path}:{i}".encode()).hexdigest(),
            "text": chunk_text,
            "file_path": rel_path,
            "package": package,
            "chunk_index": i,
            "table": table,
        }
        for i, chunk_text in enumerate(chunks_text)
    ]


# ── Embedding with reuse ──────────────────────────────────────────────────────

def embed_chunks(conn, table: str, chunks: list[dict]) -> list[list[float]]:
    """Return an embedding per chunk, calling Voyage only for text not already in the table.

    Identical texts within `chunks` are embedded once.
    """
    hashes = [content_hash(c["text"]) for c in chunks]
    known = get_embeddings_by_hash(conn, table, sorted(set(hashes)))

    pending: dict[str, str] = {}
    for h, c in zip(hashes, chunks, strict=True):
        if h not in known:
            pending.setdefault(h, c["text"])

    items = list(pending.items())
    for i in range(0, len(items), EMBED_BATCH):
        batch = items[i:i + EMBED_BATCH]
        embeddings = embed_batch([text for _, text in batch], input_type="document")
        known.update({h: emb for (h, _), emb in zip(batch, embeddings, strict=True)})

    print(f"    {len(chunks)} chunks: {len(chunks) - len(pending)} reused, {len(pending)} embedded")
    return [known[h] for h in hashes]


def index_chunks(conn, table: str, chunks: list[dict]) -> None:
    """Embed (with reuse) and upsert chunks in groups, committing per group."""
    total_groups = (len(chunks) + INDEX_GROUP - 1) // INDEX_GROUP
    for i in range(0, len(chunks), INDEX_GROUP):
        group = chunks[i:i + INDEX_GROUP]
        print(f"  Group {i // INDEX_GROUP + 1}/{total_groups}")
        upsert_chunks(conn, table, group, embed_chunks(conn, table, group))


def replace_file_chunks(
    conn, namespace: str, repo_root: Path, changed: list[str], deleted: list[str]
) -> None:
    """Re-index changed files and drop deleted ones in a single transaction.

    Embeddings are fetched before anything is deleted, so stored embeddings for
    unchanged text can be reused and the index never has a gap if embedding fails.
    """
    code_table = f"{namespace}_code"
    docs_table = f"{namespace}_docs"
    tables = [code_table, docs_table]

    all_chunks: list[dict] = []
    for rel_path in changed:
        file_path = repo_root / rel_path
        if not file_path.exists() or should_skip(Path(rel_path)):
            continue
        all_chunks.extend(chunk_file(file_path, repo_root, namespace))

    by_table = {t: [c for c in all_chunks if c["table"] == t] for t in tables}
    print(f"  New chunks: {len(all_chunks)} ({len(by_table[code_table])} code, {len(by_table[docs_table])} docs)")
    embeddings = {t: embed_chunks(conn, t, chunks) for t, chunks in by_table.items() if chunks}

    try:
        total_deleted = sum(
            delete_file_chunks(conn, tables, rel_path, commit=False) for rel_path in changed + deleted
        )
        for table, chunks in by_table.items():
            if chunks:
                upsert_chunks(conn, table, chunks, embeddings[table], commit=False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    print(f"  Removed {total_deleted} stale chunks, upserted {len(all_chunks)}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest(repo_path: str, namespace: str, repo_url: str | None = None) -> None:
    namespace = sanitise_namespace(namespace)
    code_table = f"{namespace}_code"
    docs_table = f"{namespace}_docs"
    tables = [code_table, docs_table]

    conn = get_conn(os.environ["DATABASE_URL"])
    setup_db(conn, tables)
    setup_meta_table(conn)

    repo_root = Path(repo_path).resolve()
    print(f"Scanning {repo_root} (namespace: {namespace})...")
    all_chunks: list[dict] = []

    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(repo_root)
        if should_skip(rel):
            continue
        chunks = chunk_file(file_path, repo_root, namespace)
        if chunks:
            all_chunks.extend(chunks)

    code_chunks = [c for c in all_chunks if c["table"] == code_table]
    doc_chunks = [c for c in all_chunks if c["table"] == docs_table]
    print(f"Total: {len(all_chunks)} chunks ({len(code_chunks)} code, {len(doc_chunks)} docs)")

    for table, chunks in [(code_table, code_chunks), (docs_table, doc_chunks)]:
        if not chunks:
            continue
        print(f"\nUpserting {len(chunks)} chunks to {table}...")
        index_chunks(conn, table, chunks)

    if repo_url:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo_root)
            ).decode().strip()
            register_namespace(conn, namespace, repo_url, sha)
            print(f"Registered namespace '{namespace}' at {sha[:8]}")
        except Exception as e:
            print(f"Warning: could not register namespace: {e}")

    conn.close()
    print("\nIngestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a GitHub repo into pgvector")
    parser.add_argument(
        "repo_path", nargs="?",
        help="Path to local repo checkout (clones --repo URL if omitted)",
    )
    parser.add_argument(
        "--repo", default=None,
        help="GitHub repo URL — used for cloning (when repo_path is omitted) and namespace registration",
    )
    parser.add_argument(
        "--namespace", default="tupaia",
        help="Table prefix — creates {namespace}_code and {namespace}_docs (default: tupaia)",
    )
    parser.add_argument(
        "--ref", default=None,
        help="Branch name or release tag to index (e.g. 2.50.5, v2.51.3, main). Defaults to the repo's default branch.",
    )
    args = parser.parse_args()

    if args.repo_path:
        ingest(args.repo_path, args.namespace, repo_url=args.repo)
    else:
        if not args.repo:
            parser.error("--repo URL is required when no local repo_path is given")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = os.path.join(tmpdir, "repo")
            ref_label = args.ref or "default branch"
            print(f"Cloning {args.repo} @ {ref_label} (shallow)...")
            shallow_clone(args.repo, repo_path, os.environ.get("GITHUB_TOKEN"), ref=args.ref)
            ingest(repo_path, args.namespace, repo_url=args.repo)


if __name__ == "__main__":
    main()
