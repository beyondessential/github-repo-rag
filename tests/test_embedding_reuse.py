"""Embedding reuse in scripts/ingest.py — DB and Voyage calls are faked."""

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


@pytest.fixture
def voyage(monkeypatch):
    calls: list[list[str]] = []

    def fake_embed_batch(texts, input_type="document"):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {})
    return calls


def _chunk(text: str, file_path: str = "a.py", i: int = 0) -> dict:
    return {"id": f"{file_path}:{i}", "text": text, "file_path": file_path, "package": "", "chunk_index": i,
            "table": "ns_code"}


def test_ac_009_reuses_stored_embeddings_and_dedupes_within_run(monkeypatch, voyage):
    stored = {content_hash("old"): [99.0]}
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {
        h: stored[h] for h in hashes if h in stored
    })
    chunks = [_chunk("old"), _chunk("new"), _chunk("new", "b.py"), _chunk("newer")]

    result = ingest.embed_chunks(FakeConn(), "ns_code", chunks)

    assert voyage == [["new", "newer"]]
    assert result == [[99.0], [3.0], [3.0], [5.0]]


def test_ac_009_no_voyage_call_when_everything_is_stored(monkeypatch, voyage):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {h: [1.0] for h in hashes})
    assert ingest.embed_chunks(FakeConn(), "ns_code", [_chunk("a"), _chunk("b")]) == [[1.0], [1.0]]
    assert voyage == []


def test_splits_new_texts_into_voyage_sized_batches(monkeypatch, voyage):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    chunks = [_chunk(f"t{i}", i=i) for i in range(ingest.EMBED_BATCH + 1)]
    ingest.embed_chunks(FakeConn(), "ns_code", chunks)
    assert [len(c) for c in voyage] == [ingest.EMBED_BATCH, 1]


def test_ac_011_replace_embeds_before_deleting_and_commits_once(monkeypatch, tmp_path: Path):
    (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(ingest, "embed_chunks", lambda conn, table, chunks: events.append("embed") or
                        [[0.0]] * len(chunks))
    monkeypatch.setattr(ingest, "delete_file_chunks", lambda conn, tables, path, commit: events.append(
        f"delete:{path}:{commit}") or 1)
    monkeypatch.setattr(ingest, "upsert_chunks", lambda conn, table, chunks, embs, commit: events.append(
        f"upsert:{table}:{commit}"))
    conn = FakeConn()

    ingest.replace_file_chunks(conn, "ns", tmp_path, ["a.py"], ["gone.py"])

    assert events == ["embed", "delete:a.py:False", "delete:gone.py:False", "upsert:ns_code:False"]
    assert conn.commits == 1


def test_ac_011_replace_leaves_index_untouched_when_embedding_fails(monkeypatch, tmp_path: Path):
    (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")

    def boom(conn, table, chunks):
        raise RuntimeError("voyage down")

    deletes: list[str] = []
    monkeypatch.setattr(ingest, "embed_chunks", boom)
    monkeypatch.setattr(ingest, "delete_file_chunks", lambda *a, **k: deletes.append(a[2]) or 0)

    with pytest.raises(RuntimeError):
        ingest.replace_file_chunks(FakeConn(), "ns", tmp_path, ["a.py"], [])
    assert deletes == []


def test_ac_009_uses_batch_cache_for_texts_not_in_table(monkeypatch, voyage):
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {
        h: [7.0] for h in hashes if h == content_hash("cached")
    })
    assert ingest.embed_chunks(FakeConn(), "ns_code", [_chunk("cached"), _chunk("fresh")]) == [[7.0], [5.0]]
    assert voyage == [["fresh"]]
