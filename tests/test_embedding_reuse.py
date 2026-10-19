"""Embedding reuse in scripts/ingest.py — DB and the embedding model are faked."""

from pathlib import Path

import pytest

from rag.db import content_hash
from scripts import ingest


class FakeConn:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakeModel:
    """A stand-in EmbeddingModel; embed() returns one float, len(text), per text."""

    name = "fake-model"
    max_batch = 128
    supports_batch_api = True

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts, input_type):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


@pytest.fixture
def model(monkeypatch):
    m = _FakeModel()
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {})
    return m


def _chunk(text: str, file_path: str = "a.py", i: int = 0) -> dict:
    return {"id": f"{file_path}:{i}", "text": text, "file_path": file_path, "package": "", "chunk_index": i,
            "table": "ns_code"}


def test_ac_009_reuses_stored_embeddings_and_dedupes_within_run(monkeypatch, model):
    stored = {content_hash("old"): [99.0]}
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {
        h: stored[h] for h in hashes if h in stored
    })
    chunks = [_chunk("old"), _chunk("new"), _chunk("new", "b.py"), _chunk("newer")]

    result = ingest.embed_chunks(FakeConn(), "ns_code", chunks, model)

    assert model.calls == [["new", "newer"]]
    assert result == [[99.0], [3.0], [3.0], [5.0]]


def test_ac_009_no_call_when_everything_is_stored(monkeypatch, model):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {h: [1.0] for h in hashes})
    assert ingest.embed_chunks(FakeConn(), "ns_code", [_chunk("a"), _chunk("b")], model) == [[1.0], [1.0]]
    assert model.calls == []


def test_ac_009_splits_new_texts_into_model_sized_batches(monkeypatch, model):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    chunks = [_chunk(f"t{i}", i=i) for i in range(ingest.EMBED_BATCH + 1)]
    ingest.embed_chunks(FakeConn(), "ns_code", chunks, model)
    assert [len(c) for c in model.calls] == [ingest.EMBED_BATCH, 1]


def test_ac_009_batch_size_is_also_capped_by_the_model(monkeypatch, model):
    model.max_batch = 2
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    chunks = [_chunk(f"t{i}", i=i) for i in range(5)]
    ingest.embed_chunks(FakeConn(), "ns_code", chunks, model)
    assert [len(c) for c in model.calls] == [2, 2, 1]


def test_ac_023_a_model_with_no_batch_api_never_reads_the_cache(monkeypatch, model):
    model.supports_batch_api = False

    def boom(conn, hashes):
        raise AssertionError("must not read rag_embedding_cache for a non-batch-API model")

    monkeypatch.setattr(ingest, "get_cached_embeddings", boom)
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    assert ingest.embed_chunks(FakeConn(), "ns_code", [_chunk("a")], model) == [[1.0]]


def test_ac_011_replace_embeds_before_deleting_and_commits_once(monkeypatch, tmp_path: Path, model):
    (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(ingest, "embed_chunks", lambda conn, table, chunks, m: events.append("embed") or
                        [[0.0]] * len(chunks))
    monkeypatch.setattr(ingest, "delete_file_chunks", lambda conn, tables, path, commit: events.append(
        f"delete:{path}:{commit}") or 1)
    monkeypatch.setattr(ingest, "upsert_chunks", lambda conn, table, chunks, embs, commit: events.append(
        f"upsert:{table}:{commit}"))
    conn = FakeConn()

    ingest.replace_file_chunks(conn, "ns", tmp_path, ["a.py"], ["gone.py"], model)

    assert events == ["embed", "delete:a.py:False", "delete:gone.py:False", "upsert:ns_code:False"]
    assert conn.commits == 1


def test_ac_011_replace_leaves_index_untouched_when_embedding_fails(monkeypatch, tmp_path: Path, model):
    (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")

    def boom(conn, table, chunks, m):
        raise RuntimeError("embedding down")

    deletes: list[str] = []
    monkeypatch.setattr(ingest, "embed_chunks", boom)
    monkeypatch.setattr(ingest, "delete_file_chunks", lambda *a, **k: deletes.append(a[2]) or 0)

    with pytest.raises(RuntimeError):
        ingest.replace_file_chunks(FakeConn(), "ns", tmp_path, ["a.py"], [], model)
    assert deletes == []


def test_ac_009_uses_batch_cache_for_texts_not_in_table(monkeypatch, model):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {
        h: [7.0] for h in hashes if h == content_hash("cached")
    })
    assert ingest.embed_chunks(FakeConn(), "ns_code", [_chunk("cached"), _chunk("fresh")], model) == [[7.0], [5.0]]
    assert model.calls == [["fresh"]]
