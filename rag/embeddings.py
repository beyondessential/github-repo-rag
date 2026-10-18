"""Embedding providers.

A namespace is indexed and queried with exactly one embedding model — vectors from
different models are not comparable, so a namespace never mixes them. The model a
namespace was indexed with is recorded in rag_namespaces.embedding_model (see
resolve_model() and register_namespace() in rag/db.py) and never changes implicitly.

Add a new model by instantiating VoyageModel or OpenAICompatModel into _MODELS below.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Protocol

InputType = Literal["query", "document"]

DEFAULT_MODEL_NAME = "voyage-code-3"


class EmbeddingModel(Protocol):
    name: str
    dims: int
    max_batch: int
    supports_batch_api: bool

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        """Embed texts, at most `max_batch` per call, returning one vector per text in order."""
        ...


@dataclass(frozen=True)
class VoyageModel:
    """voyage-code-3 via the Voyage AI SDK. Eligible for the Voyage Batch API."""

    name: str = "voyage-code-3"
    dims: int = 1024
    max_batch: int = 128
    supports_batch_api: bool = True

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if len(texts) > self.max_batch:
            raise ValueError(f"{self.name} accepts at most {self.max_batch} texts per call, got {len(texts)}")
        import voyageai

        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        result = client.embed(texts, model=self.name, input_type=input_type)
        return result.embeddings


@dataclass(frozen=True)
class OpenAICompatModel:
    """A model served behind an OpenAI-compatible /v1/embeddings endpoint (Ollama, Hugging
    Face TEI, vLLM). Reads EMBEDDING_BASE_URL (the endpoint's base URL, no trailing slash)
    and, if the endpoint needs one, EMBEDDING_API_KEY. Not eligible for the Voyage Batch API.
    """

    name: str
    api_model: str
    dims: int
    max_batch: int = 32
    # The exact literal prefix each model expects, prepended as-is — each model's own
    # documented format, not a shared template. Qwen3-Embedding only prefixes queries;
    # EmbeddingGemma prefixes both queries and documents, with different prefixes.
    query_prefix: str | None = None
    document_prefix: str | None = None
    supports_batch_api: bool = False

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if len(texts) > self.max_batch:
            raise ValueError(f"{self.name} accepts at most {self.max_batch} texts per call, got {len(texts)}")
        prefix = self.query_prefix if input_type == "query" else self.document_prefix
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        base_url = os.environ.get("EMBEDDING_BASE_URL")
        if not base_url:
            raise RuntimeError(f"{self.name} needs EMBEDDING_BASE_URL set to an OpenAI-compatible embeddings endpoint")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("EMBEDDING_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = json.dumps({"model": self.api_model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/embeddings", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"{self.name} embedding endpoint {base_url} returned {e.code}: {detail}") from e

        data = payload["data"]
        if all("index" in item for item in data):
            data = sorted(data, key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"{self.name} returned {len(embeddings)} embeddings for {len(texts)} inputs")
        for vec in embeddings:
            if len(vec) != self.dims:
                raise ValueError(f"{self.name} returned a {len(vec)}-dimension vector, expected {self.dims}")
        return embeddings


# Qwen3-Embedding-0.6B: Apache 2.0, 1024 dims (matches the vector(1024) column so no
# schema change is needed), strong on code retrieval for its size. Serve it with
# `ollama pull qwen3-embedding:0.6b` or Hugging Face TEI, either behind EMBEDDING_BASE_URL.
#
# EmbeddingGemma: Google, 307M params, 768 dims, Gemma Terms of Use. Built for on-device/edge
# RAG, so useful both as a lighter-weight server-side comparison point and as the leading
# candidate for an eventual on-device deployment. Serve it with `ollama pull embeddinggemma`.
_MODELS: dict[str, EmbeddingModel] = {
    "voyage-code-3": VoyageModel(),
    "qwen3-embedding-0.6b": OpenAICompatModel(
        name="qwen3-embedding-0.6b",
        api_model="qwen3-embedding:0.6b",
        dims=1024,
        # Kept small: a local Ollama server running this model under sustained load from
        # a full-repo ingest was observed returning slow HTTP 400s then hanging entirely
        # at max_batch=32. Smaller batches reduce per-request load on the server.
        max_batch=8,
        query_prefix=(
            "Instruct: Given a question about a codebase, retrieve the code or documentation "
            "that answers it\nQuery: "
        ),
    ),
    "embeddinggemma": OpenAICompatModel(
        name="embeddinggemma",
        api_model="embeddinggemma:latest",
        dims=768,
        max_batch=8,  # same conservative cap as qwen3-embedding-0.6b, unverified under load yet
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
    ),
}


def get_model(name: str) -> EmbeddingModel:
    """Look up a registered embedding model by name.

    Raises ValueError for an unknown name — there is no silent fallback, because
    guessing wrong here would index or query a namespace with the wrong model.
    """
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown embedding model {name!r}; known models: {sorted(_MODELS)}") from None
