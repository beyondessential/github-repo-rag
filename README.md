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

In HTTP mode **this server is its own OAuth authorization server**. MCP clients
(Claude Code, the claude.ai connector) register with it automatically via
Dynamic Client Registration and sign in through it; the server brokers a second
OAuth exchange with Google behind the scenes and mints its own tokens. Clients
need only the server URL — **no Google client id/secret of their own**, and no
manually pasted access token.

Configure access in `.env` (or the deploy environment):

```
# This server's externally reachable base URL — advertised as the OAuth
# authorization server via /.well-known/oauth-authorization-server, and the base
# for the Google callback redirect URI.
MCP_PUBLIC_URL=https://your-app.up.railway.app

# The server-side Google "Web application" OAuth client used to broker sign-in.
# The secret is confidential — set it in the deploy environment, never commit it.
GOOGLE_OAUTH_CLIENT_ID=123-abc.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret

# Allow anyone from your Google Workspace domain
GOOGLE_ALLOWED_DOMAIN=bes.au

# Or allowlist specific accounts
GOOGLE_ALLOWED_EMAILS=alice@example.com,bob@example.com
```

At sign-in the brokered Google token is validated against Google's tokeninfo API:
it must include the `email` scope, belong to a verified account, satisfy the
domain / email allowlist, and have been issued to `GOOGLE_OAUTH_CLIENT_ID`.
Clients then present the server's own opaque bearer token; Google tokens are kept
server-side and never handed to clients.

> **Single instance:** login state and issued tokens are held in memory, so a
> server restart makes clients re-authenticate. Move these stores to Postgres
> before scaling to more than one instance.

#### One-time Google Cloud Console setup

1. Open **APIs & Services → OAuth consent screen** and set **User type = Internal** — this restricts sign-in to your Workspace domain automatically.
2. Open **APIs & Services → Credentials → Create credentials → OAuth client ID**, type **Web application**.
3. Under **Authorised redirect URIs**, add **this server's** callback:
   `https://your-app.up.railway.app/auth/google/callback`
   (i.e. `<MCP_PUBLIC_URL>/auth/google/callback`). This is the *server's* redirect,
   not the client's — clients no longer register their own redirect with Google.
4. Copy the **Client ID** and **Client secret** into `GOOGLE_OAUTH_CLIENT_ID` /
   `GOOGLE_OAUTH_CLIENT_SECRET` in the deploy environment.

#### Connecting from claude.ai (custom connector)

1. **Settings → Connectors → Add custom connector**.
2. **Remote MCP server URL**: `https://your-app.up.railway.app/mcp` (the full `/mcp` path).
3. **Connect** — no Advanced settings / client id needed. The connector registers
   automatically and redirects you to Google to sign in with a Workspace account.

#### Connecting from Claude Code (HTTP server)

Register the server with just its URL — Claude Code discovers OAuth, registers,
and opens a browser to sign in (complete it with `/mcp`):

```json
{
  "mcpServers": {
    "github-repo-rag": {
      "type": "http",
      "url": "https://your-app.up.railway.app/mcp"
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

`scripts/ingest.py --repo` registers its namespace automatically, so anything indexed via [Indexing a repository](#indexing-a-repository) is already eligible for sync. `scripts/sync.py` then keeps every registered namespace up to date: for each one it compares the last-indexed commit SHA against the latest on the default branch, re-embeds only the changed files, and deletes chunks for removed files. It falls back to a full reindex when the diff is too large for GitHub's compare API (> 250 commits or ≥ 300 files changed).

```bash
uv run python scripts/sync.py                     # sync all registered namespaces
uv run python scripts/sync.py --namespace tupaia   # sync one namespace
```

In production this runs on a schedule via a Railway cron service — see [Deployment](#deployment).

Two lower-level scripts back this:
- `scripts/register.py` — register (or re-register) a namespace's baseline commit SHA without re-ingesting. Useful after a manual `ingest.py` run, or to clear the SHA and force a full reindex on the next sync.
- `scripts/reindex.py` — re-embed an explicit set of changed/deleted files, given as `CHANGED_FILES` / `DELETED_FILES` (space-separated repo-relative paths). This is the primitive `sync.py` builds on; call it directly only if you already know exactly which files changed.

## Deployment

The MCP server and the scheduled sync run as two separate services in the same Railway project, sharing the internal Postgres instance.

### MCP server

Deploys from [`railway.toml`](railway.toml) (Dockerfile build). Needs the base `DATABASE_URL` / `VOYAGE_API_KEY` from [Setup](#setup), plus the HTTP-mode env vars from [MCP over HTTP](#mcp-over-http-shared-team-server), in the service's Variables tab.

### Scheduled sync (cron service)

The daily incremental sync runs as a separate Railway service (not defined in a repo file — configured directly in the Railway dashboard):

1. In the same Railway project, **+ New → Empty Service**, connected to this repo so it builds from the same `Dockerfile` (which copies `scripts/` into the image).
2. **Settings → Deploy → Custom Start Command:**
   ```
   uv run python scripts/sync.py
   ```
3. **Settings → Deploy → Cron Schedule:**
   ```
   0 15 * * *
   ```
   Setting a cron schedule makes Railway run the container to completion on that schedule instead of keeping it up as a long-lived service.
4. **Variables:**
   - `DATABASE_URL` — reference the **internal** URL from the Postgres service (e.g. `${{Postgres.DATABASE_URL}}`), not `DATABASE_PUBLIC_URL`. This is why the cron service lives inside the Railway project: the sync job runs over the private network, so Postgres never needs a public port.
   - `VOYAGE_API_KEY` — Voyage AI key.
   - `GITHUB_TOKEN` — optional; raises the GitHub API rate limit from 60 to 5000 req/hr.
5. Deploy, then check **Deployments** to confirm it runs once per schedule rather than continuously.

For an ad-hoc full reindex of one namespace, override the start command on that service for a single run, or run locally:
```bash
uv run python scripts/ingest.py --repo <repo_url> --namespace <namespace>
```

## Code review

Pull requests are automatically reviewed by Claude via `.github/workflows/claude-code-review.yml`, which delegates to the shared [`maui-team`](https://github.com/beyondessential/maui-team) workflow. Re-trigger a review by commenting `/review` on any PR.

Requires `ANTHROPIC_API_KEY` set as a repository secret.
