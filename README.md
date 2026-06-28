# github-repo-rag

Generic RAG pipeline for GitHub repositories. Index any repo into PostgreSQL (pgvector) and query it via hybrid search — from Python, or from any Claude agent via MCP.

## How it works

```
GitHub repo
    ↓  scripts/ingest.py
    chunk by file/function (langchain-text-splitters)
    embed with Voyage AI or Ollama (configurable)
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
- One of:
  - [Voyage AI](https://www.voyageai.com) API key (default, cloud embeddings)
  - [Ollama](https://ollama.com) running locally (free, local embeddings)

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

**.env** (Voyage AI — default)
```
DATABASE_URL=postgresql://user:password@localhost:5432/rag
VOYAGE_API_KEY=your-voyage-key
```

**.env** (Ollama — local)
```
DATABASE_URL=postgresql://user:password@localhost:5432/rag
EMBED_BACKEND=ollama
# OLLAMA_HOST=http://localhost:11434   # default
# OLLAMA_EMBED_MODEL=mxbai-embed-large  # default, 1024 dims
```

When using Ollama, pull the embedding model first:
```bash
ollama pull mxbai-embed-large
```

## Indexing a repository

```bash
# Clone and index automatically (default branch)
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --namespace tupaia

# Or point at a local checkout
uv run python scripts/ingest.py /path/to/tupaia --namespace tupaia
```

This creates `tupaia_code` and `tupaia_docs` tables in your database. Re-run at any time to refresh; existing chunks are upserted.

### Indexing a specific release or branch

Use `--ref` to index a specific release tag or branch instead of the default branch:

```bash
# Index a specific release tag
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref 2.50.5 --namespace tupaia

# Index a specific branch
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref main --namespace tupaia
```

To keep multiple versions queryable simultaneously, use a different namespace per version:

```bash
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref 2.50.5 --namespace tupaia_2_50
uv run python scripts/ingest.py --repo https://github.com/beyondessential/tupaia --ref 2.51.3 --namespace tupaia_2_51
```

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

#### Deploying to Railway

The repo includes a `Dockerfile` and `railway.toml` for one-command deploys. The database is already hosted on Railway, so deploying the MCP server there keeps everything on the same internal network.

1. Install the [Railway CLI](https://docs.railway.com/guides/cli) and log in:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. Link to your existing project (the one with the Postgres database) and create a service:
   ```bash
   cd github-repo-rag
   railway link
   railway service create github-repo-rag-mcp
   railway service   # select the new service
   ```

3. Set environment variables:
   ```bash
   railway variables set DATABASE_URL='${{Postgres.DATABASE_URL}}'
   railway variables set VOYAGE_API_KEY=<your-voyage-key>   # or set EMBED_BACKEND=ollama + OLLAMA_HOST
   railway variables set MCP_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   ```
   The `${{Postgres.DATABASE_URL}}` reference uses Railway's internal networking (faster, no public internet hop).

4. Deploy and generate a public URL:
   ```bash
   railway up
   railway domain
   ```

The MCP endpoint will be at `https://<your-railway-domain>/mcp`. A health check is available at `/health`.

#### Authentication

When `MCP_API_KEY` is set, all requests to `/mcp` require a matching `Authorization: Bearer <key>` header. The `/health` endpoint is always public. If `MCP_API_KEY` is not set, the server runs without authentication.

**Connecting from Claude Code** (shared HTTP server):
```json
{
  "mcpServers": {
    "github-repo-rag": {
      "type": "url",
      "url": "https://<your-railway-domain>/mcp",
      "headers": {
        "Authorization": "Bearer <your-mcp-api-key>"
      }
    }
  }
}
```

Add this to `~/.claude/settings.json` (global) or `.mcp.json` (per-project).

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

For ad-hoc querying from the terminal, `scripts/ask.py` runs the full pipeline locally. Set `LLM_BACKEND` to choose the answering LLM:

- `anthropic` (default) — requires `ANTHROPIC_API_KEY`
- `ollama` — requires Ollama running with a chat model (set `OLLAMA_CHAT_MODEL`, default: `llama3.1`)

```bash
uv run python scripts/ask.py "How does survey response validation work?"
uv run python scripts/ask.py --namespace tamanu "How are encounters structured?"

# Use Ollama for answering
LLM_BACKEND=ollama uv run python scripts/ask.py "How does survey response validation work?"
```

## Incremental reindex

For keeping an index up to date after changes, set `CHANGED_FILES` and `DELETED_FILES` (space-separated repo-relative paths) and run:

```bash
CHANGED_FILES="src/foo.ts src/bar.ts" DELETED_FILES="src/old.ts" \
  uv run python scripts/reindex.py /path/to/repo --namespace tupaia
```

The GitHub Actions workflow (`.github/workflows/reindex.yml`) runs a full reindex every Monday and can be triggered manually via `workflow_dispatch`.

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
.env.example
```
