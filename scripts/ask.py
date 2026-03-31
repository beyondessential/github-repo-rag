#!/usr/bin/env python3
"""
Local CLI for querying the RAG pipeline end-to-end (retrieve + answer with Claude).

This is a developer convenience script — not part of the MCP server.
The MCP server only exposes retrieval; the calling LLM does the answering.

Usage:
    python scripts/ask.py "How does the data-broker package work?"
    python scripts/ask.py --namespace tamanu "How are surveys submitted?"

Environment variables:
    DATABASE_URL       — PostgreSQL connection string
    VOYAGE_API_KEY     — Voyage AI API key
    ANTHROPIC_API_KEY  — Anthropic API key
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
from rag.query import retrieve

_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use the retrieved context below to answer questions accurately. "
    "Your audience is non-technical — explain concepts in plain language, avoid jargon, and focus on what things do rather than how they are implemented. "
    "If the user asks for technical details, you may include file paths, package names, and implementation specifics. "
    "If the retrieved context doesn't fully answer the question, say what you know and be honest about gaps."
)


def answer(
    question: str,
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    namespace: str = "tupaia",
) -> str:
    tables = [f"{namespace}_code", f"{namespace}_docs"]
    context = retrieve(question, tables=tables)
    prefix = system_prompt or _DEFAULT_SYSTEM
    system = f"{prefix}\n\n## Retrieved context\n\n{context}"

    messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
    messages.append({"role": "user", "content": question})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline with Claude")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--namespace", default="tupaia",
        help="Indexed repo namespace to search (default: tupaia)",
    )
    args = parser.parse_args()

    print(answer(args.question, namespace=args.namespace))


if __name__ == "__main__":
    main()
