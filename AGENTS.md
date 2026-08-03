@./.maui/knowledge/AGENT.base.md
@./.maui/knowledge/standards/git-conventions.md
@./.maui/knowledge/standards/python-conventions.md

# github-repo-rag

Generic RAG pipeline for GitHub repositories — ingest, index, and query any repo via pgvector. Exposes search as an MCP tool consumed by Claude Code and other AI clients.

## Architecture

```
Ingest:  GitHub repo → chunk (langchain-text-splitters) → embed (voyage-code-3) → upsert (pgvector)
Query:   question → embed → hybrid search (vector + FTS with RRF) → context chunks
MCP:     search_codebase / get_file / list_files / get_repo_structure / list_namespaces / namespace_info → returns raw context (client LLM answers)
```

The MCP server does **retrieval only** — it never calls an LLM. The calling agent (Claude Code, Cursor, etc.) does the answering using the returned context chunks.

## Key files

- `mcp_server.py` — FastMCP server; exposes `search_codebase`, `get_file`, `list_files`, `get_repo_structure`, `list_namespaces`, `namespace_info` tools
- `rag/query.py` — `embed()` and `retrieve()` (hybrid pgvector search)
- `rag/db.py` — DB setup and upsert helpers
- `rag/auth.py` — Google OAuth token verifier for HTTP transport
- `scripts/ingest.py` — ingest a repo into pgvector
- `scripts/ask.py` — local CLI for full retrieve + answer pipeline (uses Anthropic API; dev convenience only)
- `scripts/reindex.py` — drop and re-ingest a namespace

## Transport modes

- **stdio** (default) — local subprocess, no auth; for Claude Code / local agents
- **http** — network server with optional Google OAuth; for shared team use

## Namespaces

Each indexed repo gets a namespace (e.g. `tupaia`), creating `{namespace}_code` and `{namespace}_docs` tables. Namespace names are sanitised to valid PostgreSQL identifiers (e.g. `data-lake` → `data_lake`). Pass `namespace` to any tool to target the right repo.

## Environment variables

| Variable | Used by |
|---|---|
| `DATABASE_URL` | All DB operations |
| `VOYAGE_API_KEY` | Embedding (ingest + query) |
| `ANTHROPIC_API_KEY` | `scripts/ask.py` only — not required for MCP |
| `GOOGLE_ALLOWED_DOMAIN` / `GOOGLE_ALLOWED_EMAILS` | HTTP transport auth (optional) |

## Conventions

- Python 3.11+
- Dependencies managed via `pyproject.toml` / `uv`
- `anthropic` is **not** a core dependency — it is only needed for `scripts/ask.py`
- SQL in `retrieve()` uses parameterised queries; never interpolate user input directly
