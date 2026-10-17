"""rag/embeddings.py — the Voyage SDK and the OpenAI-compatible HTTP endpoint are faked."""

import io
import json
import urllib.error

import pytest

from rag import embeddings as em

# ── get_model ─────────────────────────────────────────────────────────────────

def test_get_model_returns_the_registered_model():
    model = em.get_model("voyage-code-3")
    assert model.name == "voyage-code-3"
    assert model.dims == 1024
    assert model.supports_batch_api is True


def test_get_model_raises_on_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown embedding model 'bogus'"):
        em.get_model("bogus")


def test_default_model_is_registered():
    assert em.get_model(em.DEFAULT_MODEL_NAME).name == em.DEFAULT_MODEL_NAME


# ── VoyageModel ───────────────────────────────────────────────────────────────

@pytest.fixture
def fake_voyage(monkeypatch):
    calls: list[dict] = []

    class _Result:
        def __init__(self, n: int) -> None:
            self.embeddings = [[float(i)] for i in range(n)]

    class _FakeClient:
        def __init__(self, api_key) -> None:
            calls.append({"api_key": api_key})

        def embed(self, texts, model, input_type):
            calls.append({"texts": list(texts), "model": model, "input_type": input_type})
            return _Result(len(texts))

    monkeypatch.setattr("voyageai.Client", _FakeClient)
    monkeypatch.setenv("VOYAGE_API_KEY", "key-123")
    return calls


def test_voyage_model_embeds_with_its_own_model_name_and_input_type(fake_voyage):
    model = em.VoyageModel()
    result = model.embed(["a", "b"], "document")
    assert result == [[0.0], [1.0]]
    assert fake_voyage[1] == {"texts": ["a", "b"], "model": "voyage-code-3", "input_type": "document"}


def test_voyage_model_rejects_a_batch_larger_than_max_batch(fake_voyage):
    model = em.VoyageModel(max_batch=2)
    with pytest.raises(ValueError, match="at most 2 texts"):
        model.embed(["a", "b", "c"], "document")


# ── OpenAICompatModel ─────────────────────────────────────────────────────────

@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    return em.OpenAICompatModel(
        name="qwen3-embedding-0.6b", api_model="qwen3-embedding:0.6b", dims=2, max_batch=4,
        query_instruction="Find the code that answers this question",
    )


def _fake_urlopen(response_body: dict, capture: dict):
    class _Resp:
        def read(self) -> bytes:
            return json.dumps(response_body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    def urlopen(req):
        capture["url"] = req.full_url
        capture["headers"] = dict(req.headers)
        capture["body"] = json.loads(req.data)
        return _Resp()

    return urlopen


def test_openai_compat_model_posts_to_the_base_url(monkeypatch, model):
    capture: dict = {}
    data = [{"embedding": [1.0, 2.0], "index": 0}, {"embedding": [3.0, 4.0], "index": 1}]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen({"data": data}, capture))
    result = model.embed(["a", "b"], "document")
    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert capture["url"] == "http://localhost:11434/v1/embeddings"
    assert capture["body"] == {"model": "qwen3-embedding:0.6b", "input": ["a", "b"]}
    assert "Authorization" not in capture["headers"]


def test_openai_compat_model_adds_the_api_key_header_when_set(monkeypatch, model):
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    capture: dict = {}
    monkeypatch.setattr(
        "urllib.request.urlopen", _fake_urlopen({"data": [{"embedding": [1.0, 2.0], "index": 0}]}, capture)
    )
    model.embed(["a"], "document")
    assert capture["headers"]["Authorization"] == "Bearer secret"


def test_openai_compat_model_prefixes_queries_but_not_documents(monkeypatch, model):
    capture: dict = {}
    monkeypatch.setattr(
        "urllib.request.urlopen", _fake_urlopen({"data": [{"embedding": [1.0, 2.0], "index": 0}]}, capture)
    )
    model.embed(["how does X work?"], "query")
    assert capture["body"]["input"] == [
        "Instruct: Find the code that answers this question\nQuery: how does X work?"
    ]

    model.embed(["how does X work?"], "document")
    assert capture["body"]["input"] == ["how does X work?"]


def test_openai_compat_model_reorders_by_index(monkeypatch, model):
    capture: dict = {}
    data = [{"embedding": [9.0, 9.0], "index": 1}, {"embedding": [1.0, 1.0], "index": 0}]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen({"data": data}, capture))
    assert model.embed(["first", "second"], "document") == [[1.0, 1.0], [9.0, 9.0]]


def test_openai_compat_model_raises_on_a_dimension_mismatch(monkeypatch, model):
    capture: dict = {}
    monkeypatch.setattr(
        "urllib.request.urlopen", _fake_urlopen({"data": [{"embedding": [1.0, 2.0, 3.0], "index": 0}]}, capture)
    )
    with pytest.raises(ValueError, match="3-dimension vector, expected 2"):
        model.embed(["a"], "document")


def test_openai_compat_model_raises_on_an_http_error(monkeypatch, model):
    def urlopen(req):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"server exploded"))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(RuntimeError, match="returned 500"):
        model.embed(["a"], "document")


def test_openai_compat_model_rejects_a_batch_larger_than_max_batch(model):
    with pytest.raises(ValueError, match="at most 4 texts"):
        model.embed(["a", "b", "c", "d", "e"], "document")


def test_openai_compat_model_requires_embedding_base_url(monkeypatch, model):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        model.embed(["a"], "document")
