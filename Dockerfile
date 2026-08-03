FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen = respect lockfile exactly)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY mcp_server.py ./
COPY rag/ ./rag/
# scripts/ powers the scheduled reindex (Railway cron service runs scripts/sync.py)
COPY scripts/ ./scripts/

# Railway injects PORT; default to 8765 for local testing
ENV PORT=8765

EXPOSE ${PORT}

CMD uv run python mcp_server.py --transport http --host 0.0.0.0 --port ${PORT}
