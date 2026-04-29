#!/usr/bin/env python3
"""
MCP server for github-repo-rag.

Two transport modes:
  stdio (default) — local subprocess, no auth needed.
  http            — network server, optional API key auth.

Usage:
    # Local (Claude Code / local agents):
    uv run python mcp_server.py

    # Network (shared team server):
    uv run python mcp_server.py --transport http --host 0.0.0.0 --port 8765

Register in .mcp.json or ~/.claude/settings.json — see README.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load credentials from this repo's .env so no env vars are needed in .mcp.json
load_dotenv(Path(__file__).parent / ".env")


# ── MCP server factory ─────────────────────────────────────────────────────────

def _build_mcp(host: str, port: int):
    """Build a FastMCP instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("github-repo-rag", host=host, port=port)

    # ── Health check ──────────────────────────────────────────────────────────

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ── Tools ──────────────────────────────────────────────────────────────────

    from rag.db import sanitise_namespace

    @mcp.tool()
    def search_codebase(question: str, namespace: str = "tupaia") -> str:
        """Search an indexed GitHub repository and return relevant source code and documentation.

        Use this tool whenever a question involves understanding how a codebase works, finding
        where a feature is implemented, or answering questions about specific packages, modules,
        or patterns in an indexed repo. Call list_namespaces() first if you are unsure which
        namespace to use.

        The tool performs hybrid vector + full-text search (RRF) and returns raw chunks — you
        must synthesise the answer from the returned context.

        Args:
            question:  Natural language question, or a specific symbol/concept to look up
                       (e.g. "How does the data-broker package work?", "PatientForm component").
            namespace: Identifier for the indexed repo to search (e.g. "tupaia"). Defaults to
                       "tupaia". Use list_namespaces() to see all available repos.
        """
        from rag.query import retrieve

        namespace = sanitise_namespace(namespace)
        tables = [f"{namespace}_code", f"{namespace}_docs"]
        return retrieve(question, tables=tables)

    @mcp.tool()
    def list_namespaces() -> list[str]:
        """List all indexed repositories available for search.

        Returns namespace names that can be passed to search_codebase().
        """
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name LIKE '%_code'
                ORDER BY table_name
            """)
            rows = cur.fetchall()
        conn.close()
        return [row[0].removesuffix("_code") for row in rows]

    @mcp.tool()
    def get_file(file_path: str, namespace: str = "tupaia") -> str:
        """Return the content of a specific file from an indexed repository.

        Use this after search_codebase() identifies a relevant file and you need
        to read its complete content rather than just the matching chunks.

        Args:
            file_path: Relative path within the repo (e.g. "rag/query.py").
            namespace:  Namespace of the indexed repo. Use list_namespaces() to see options.
        """
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(conn)

        namespace = sanitise_namespace(namespace)
        chunks: list[str] = []
        for table in [f"{namespace}_code", f"{namespace}_docs"]:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT text FROM {table} WHERE file_path = %s ORDER BY chunk_index",
                        (file_path,),
                    )
                    chunks.extend(row[0] for row in cur.fetchall())
            except Exception:
                pass

        conn.close()

        if not chunks:
            return f"(No content found for '{file_path}' in namespace '{namespace}')"

        return f"# {file_path}\n\n" + "\n\n---\n\n".join(chunks)

    @mcp.tool()
    def namespace_info(namespace: str = "tupaia") -> dict:
        """Return metadata about an indexed namespace.

        Useful for checking ingestion health: how many files and chunks were indexed,
        when it was last indexed, and at which commit.

        Args:
            namespace: Namespace to inspect. Use list_namespaces() to see options.
        """
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(conn)

        namespace = sanitise_namespace(namespace)

        def table_stats(table: str) -> tuple[int, int]:
            """Return (file_count, chunk_count) for a table, or (0, 0) if missing."""
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT COUNT(DISTINCT file_path), COUNT(*) FROM {table}"
                    )
                    return cur.fetchone()
            except Exception:
                return 0, 0

        code_files, code_chunks = table_stats(f"{namespace}_code")
        doc_files, doc_chunks = table_stats(f"{namespace}_docs")

        registry: dict = {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT repo_url, last_commit_sha, last_indexed_at "
                    "FROM rag_namespaces WHERE namespace = %s",
                    (namespace,),
                )
                row = cur.fetchone()
                if row:
                    registry = {
                        "repo_url": row[0],
                        "last_commit_sha": row[1],
                        "last_indexed_at": row[2].isoformat() if row[2] else None,
                    }
        except Exception:
            pass

        conn.close()

        return {
            "namespace": namespace,
            **registry,
            "code_files": code_files,
            "code_chunks": code_chunks,
            "doc_files": doc_files,
            "doc_chunks": doc_chunks,
            "total_chunks": code_chunks + doc_chunks,
        }

    @mcp.tool()
    def list_files(namespace: str = "tupaia", prefix: str = "") -> list[str]:
        """List all indexed file paths in a namespace, optionally filtered by path prefix.

        Use this to explore what files are available before calling get_file(), or to
        find files in a specific directory.

        Args:
            namespace: Namespace of the indexed repo. Use list_namespaces() to see options.
            prefix:    Optional path prefix to filter results (e.g. "rag/" or "scripts/").
        """
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(conn)

        namespace = sanitise_namespace(namespace)
        paths: set[str] = set()
        for table in [f"{namespace}_code", f"{namespace}_docs"]:
            try:
                with conn.cursor() as cur:
                    if prefix:
                        cur.execute(
                            f"SELECT DISTINCT file_path FROM {table} WHERE file_path LIKE %s ORDER BY file_path",
                            (f"{prefix}%",),
                        )
                    else:
                        cur.execute(f"SELECT DISTINCT file_path FROM {table} ORDER BY file_path")
                    paths.update(row[0] for row in cur.fetchall())
            except Exception:
                pass

        conn.close()
        return sorted(paths)

    @mcp.tool()
    def get_repo_structure(namespace: str = "tupaia", depth: int = 2) -> list[str]:
        """Return the directory tree of an indexed repository up to a given depth.

        Useful for understanding the high-level layout of a repo before diving into
        specific files or directories.

        Args:
            namespace: Namespace of the indexed repo. Use list_namespaces() to see options.
            depth:     How many directory levels to include (default: 2). Use 1 for top-level
                       directories only, or higher values for more detail.
        """
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        register_vector(conn)

        namespace = sanitise_namespace(namespace)
        all_paths: set[str] = set()
        for table in [f"{namespace}_code", f"{namespace}_docs"]:
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT DISTINCT file_path FROM {table}")
                    all_paths.update(row[0] for row in cur.fetchall())
            except Exception:
                pass

        conn.close()

        dirs: set[str] = set()
        for path in all_paths:
            parts = path.split("/")
            # Add each ancestor directory up to `depth` levels
            for i in range(1, min(len(parts), depth + 1)):
                dirs.add("/".join(parts[:i]) + "/")
            # Also include the file itself if it's at or within the depth
            if len(parts) <= depth + 1:
                dirs.add(path)

        return sorted(dirs)

    return mcp


# ── API key middleware ────────────────────────────────────────────────────────

def _wrap_with_api_key_auth(app):
    """Wrap a Starlette app with API key verification.

    Checks for MCP_API_KEY env var. If set, requires all requests (except
    /health) to include a matching Authorization: Bearer <key> header.
    """
    from starlette.responses import JSONResponse

    api_key = os.environ.get("MCP_API_KEY", "")

    async def middleware(scope, receive, send):
        if scope["type"] == "http" and api_key:
            from starlette.requests import Request

            request = Request(scope, receive, send)
            # Let health check through without auth
            if request.url.path != "/health":
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer ") or auth_header[7:] != api_key:
                    response = JSONResponse(
                        {"error": "unauthorized", "message": "Invalid or missing API key"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    await response(scope, receive, send)
                    return
        await app(scope, receive, send)

    return middleware


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="github-repo-rag MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport: stdio (default, local) or http (network)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind in HTTP mode (default: 127.0.0.1; use 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port in HTTP mode (default: 8765)",
    )
    args = parser.parse_args()

    mcp = _build_mcp(host=args.host, port=args.port)

    if args.transport == "http":
        import uvicorn

        # Build the Starlette app, optionally wrapped with API key auth
        starlette_app = mcp.streamable_http_app()
        if os.environ.get("MCP_API_KEY"):
            starlette_app = _wrap_with_api_key_auth(starlette_app)
            print("API key auth enabled (MCP_API_KEY is set)")
        else:
            print(
                "WARNING: HTTP mode started without auth. "
                "Set MCP_API_KEY in env vars to restrict access."
            )

        print(f"Starting MCP server on http://{args.host}:{args.port}/mcp")
        config = uvicorn.Config(
            starlette_app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        import anyio
        anyio.run(server.serve)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
