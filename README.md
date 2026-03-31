# github-repo-rag

Generic RAG pipeline for GitHub repositories. Index any repo into PostgreSQL (pgvector) and query it via hybrid search — from Python, or from any Claude agent via MCP.

## How it works

```
GitHub repo
    ↓  scripts/ingest.py
    chunk by file/function (langchain-text-splitters)
    embed with voyage-code-3 (Voyage AI)
    upsert to PostgreSQL + pgvector
          ↑
    hybrid search at query time
    (vector similarity + full-text search, RRF fusion)
          ↑
    MCP tools (search_codebase, get_file, list_files, get_repo_structure, list_namespaces, namespace_info)
          ↑
    Claude agent (does the answering)
```

Two tables are created per namespace: `{namespace}_code` (source files, ~512 token chunks) and `{namespace}_docs` (READMEs and docs, ~1024 token chunks).

The MCP server does **retrieval only** — it returns context chunks to the calling agent. The agent does the answering using its own LLM. No Anthropic API key is required to run the server.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension enabled
- [Voyage AI](https://www.voyageai.com) API key

Enable pgvector on your database (run once):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Setup

```bash
git clone --recurse-submodules https://github.com/your-org/github-repo-rag
cd github-repo-rag
uv sync
cp .env.example .env
# edit .env with your credentials
```

**.env**
```
DATABASE_URL=postgresql://user:password@localhost:5432/rag
VOYAGE_API_KEY=your-voyage-key
```

## Indexing a repository

```bash
# Clone and index automatically
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --namespace tupaia

# Or point at a local checkout
uv run python scripts/ingest.py /path/to/tupaia --namespace tupaia
```

This creates `tupaia_code` and `tupaia_docs` tables in your database. Re-run at any time to refresh; existing chunks are upserted.

To index a different repo, change `--repo` and `--namespace`:
```bash
uv run python scripts/ingest.py --repo https://github.com/org/myrepo --namespace myrepo
```

## Querying

### MCP tool (Claude Code / Claude agents)

The MCP server exposes six tools:

| Tool | Description |
|------|-------------|
| `search_codebase(question, namespace)` | Hybrid vector + FTS search; returns relevant context chunks |
| `get_file(file_path, namespace)` | Return the full content of a specific file in order |
| `list_files(namespace, prefix)` | List all indexed file paths, optionally filtered by path prefix |
| `get_repo_structure(namespace, depth)` | Directory tree up to a given depth |
| `list_namespaces()` | List all indexed repos available in the database |
| `namespace_info(namespace)` | Ingestion health: file/chunk counts, last commit, last indexed time |

**Register for a specific project** — add `.mcp.json` to the project root:
```json
{
  "mcpServers": {
    "github-repo-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/github-repo-rag", "github-repo-rag-mcp"]
    }
  }
}
```

**Register globally** — add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "github-repo-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/github-repo-rag", "github-repo-rag-mcp"]
    }
  }
}
```

Once registered, Claude Code can call `search_codebase` directly as a tool during any conversation.

### MCP over HTTP (shared team server)

Run the server in HTTP mode so multiple agents and team members can share one instance:

```bash
uv run python mcp_server.py --transport http --host 0.0.0.0 --port 8765
```

In HTTP mode the server requires a valid Google OAuth2 access token on every request.
Configure access restrictions in `.env`:

```
# Allow anyone from your Google Workspace domain
GOOGLE_ALLOWED_DOMAIN=bes.au

# Or allowlist specific accounts
GOOGLE_ALLOWED_EMAILS=alice@example.com,bob@example.com
```

Tokens are validated against Google's tokeninfo API. The token must include the `email` scope and belong to a verified Google account.

**Getting a token** (CLI):
```bash
gcloud auth print-access-token
```

**Connecting from Claude Code** (HTTP server):
```json
{
  "mcpServers": {
    "github-repo-rag": {
      "url": "http://your-server:8765/mcp",
      "headers": {
        "Authorization": "Bearer <google-access-token>"
      }
    }
  }
}
```

The `stdio` transport (default) is for local use only and does not require authentication.

### Python API

```python
from rag.query import retrieve

# Retrieve context chunks (returns a formatted string)
context = retrieve("How does survey response validation work?", tables=["tupaia_code", "tupaia_docs"])
```

Install as a path dependency in another project (via uv):
```toml
# pyproject.toml
dependencies = ["github-repo-rag"]

[tool.uv.sources]
github-repo-rag = { path = "../github-repo-rag", editable = true }
```

### Local CLI (retrieve + answer)

For ad-hoc querying from the terminal, `scripts/ask.py` runs the full pipeline locally using Claude. Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
uv run python scripts/ask.py "How does survey response validation work?"
uv run python scripts/ask.py --namespace tamanu "How are encounters structured?"
```

## Incremental reindex

For keeping an index up to date after changes, set `CHANGED_FILES` and `DELETED_FILES` (space-separated repo-relative paths) and run:

```bash
CHANGED_FILES="src/foo.ts src/bar.ts" DELETED_FILES="src/old.ts" \
  uv run python scripts/reindex.py /path/to/repo --namespace tupaia
```

The GitHub Actions workflow (`.github/workflows/reindex.yml`) runs a full reindex every Monday and can be triggered manually via `workflow_dispatch`.

## Code review

Pull requests are automatically reviewed by Claude via `.github/workflows/claude-code-review.yml`, which delegates to the shared [`maui-team`](https://github.com/beyondessential/maui-team) workflow. Re-trigger a review by commenting `/review` on any PR.

Requires `ANTHROPIC_API_KEY` set as a repository secret.

## File structure

```
mcp_server.py          # MCP server (search_codebase, get_file, list_files, get_repo_structure, list_namespaces, namespace_info)
rag/
  query.py             # embed(), retrieve()
  db.py                # setup_db(), upsert_chunks(), delete_file_chunks()
  auth.py              # Google OAuth token verifier (HTTP transport)
scripts/
  ingest.py            # full ingestion CLI
  reindex.py           # incremental reindex CLI
  ask.py               # local CLI: retrieve + answer with Claude
.maui/                 # submodule: shared AI knowledge and reusable workflows
AGENTS.md              # AI agent context (imports from .maui/knowledge/)
.github/
  workflows/
    reindex.yml        # weekly GitHub Actions reindex
    claude-code-review.yml  # automated PR review via Claude
.env.example
```
