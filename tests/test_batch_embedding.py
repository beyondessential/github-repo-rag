"""Voyage Batch API flow — specs/pipelines/batch-embedding.md. DB, Voyage and git are faked."""

import json
from pathlib import Path

import pytest

from rag import voyage_batch
from scripts import ingest, sync


class FakeConn:
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def db(monkeypatch):
    """Stub every DB call ingest() makes and record what it does."""
    calls: dict[str, list] = {"upserts": [], "batches": [], "registered": [], "events": []}
    monkeypatch.setattr(ingest, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(ingest, "setup_db", lambda conn, tables: None)
    monkeypatch.setattr(ingest, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(ingest, "get_existing_hashes", lambda conn, tables, hashes: set())
    monkeypatch.setattr(ingest, "get_embeddings_by_hash", lambda conn, table, hashes: {})
    monkeypatch.setattr(ingest, "get_cached_embeddings", lambda conn, hashes: {})
    monkeypatch.setattr(ingest, "embed_batch", lambda texts, input_type: [[0.0]] * len(texts))
    def upsert(conn, table, chunks, embs):
        calls["upserts"].append(table)
        calls["events"].append(f"upsert:{table}")

    def delete_except(conn, table, keep_ids):
        calls["events"].append(f"prune:{table}:{len(keep_ids)}")
        return 0

    monkeypatch.setattr(ingest, "upsert_chunks", upsert)
    monkeypatch.setattr(ingest, "delete_chunks_except", delete_except)
    monkeypatch.setattr(ingest, "add_pending_batch", lambda conn, b, ns: calls["batches"].append((b, ns)))
    monkeypatch.setattr(ingest, "register_namespace", lambda conn, ns, url, sha: calls["registered"].append(sha))
    monkeypatch.setattr(voyage_batch, "upload_file", lambda content, name: "file-1")
    monkeypatch.setattr(voyage_batch, "create_batch", lambda file_id, metadata: {"id": "batch-1"})
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    return calls


def test_ac_001_below_threshold_embeds_in_real_time(monkeypatch, repo, db):
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "6")
    assert ingest.ingest(str(repo), "ns", use_batch=True) is True
    assert db["batches"] == []
    assert db["upserts"] == ["ns_code"]


def test_ac_002_at_threshold_submits_batch_and_upserts_nothing(monkeypatch, repo, db):
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "5")
    assert ingest.ingest(str(repo), "ns", repo_url="https://github.com/o/r", use_batch=True) is False
    assert db["batches"] == [("batch-1", "ns")]
    assert db["upserts"] == []
    assert db["registered"] == [None]


def test_ac_002_batching_off_by_default(monkeypatch, repo, db):
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "0")
    assert ingest.ingest(str(repo), "ns") is True
    assert db["batches"] == []


def test_full_reindex_prunes_stale_chunks_after_all_upserts(monkeypatch, repo, db):
    (repo / "README.md").write_text("# Docs\n", encoding="utf-8")
    assert ingest.ingest(str(repo), "ns") is True
    assert db["events"] == ["upsert:ns_code", "upsert:ns_docs", "prune:ns_code:5", "prune:ns_docs:1"]


def test_submitted_batch_prunes_nothing(monkeypatch, repo, db):
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "1")
    assert ingest.ingest(str(repo), "ns", repo_url="https://github.com/o/r", use_batch=True) is False
    assert db["events"] == []


def test_empty_checkout_refuses_to_reindex(tmp_path, db):
    with pytest.raises(RuntimeError, match="No indexable files"):
        ingest.ingest(str(tmp_path), "ns")
    assert db["events"] == []


def test_invalid_threshold_fails_loudly(monkeypatch):
    monkeypatch.setenv("VOYAGE_BATCH_THRESHOLD", "lots")
    with pytest.raises(SystemExit):
        ingest.batch_threshold()


def test_ac_003_input_lines_keyed_by_hash_and_split_at_caps(monkeypatch):
    monkeypatch.setattr(voyage_batch, "MAX_LINES", 2)
    files = voyage_batch.build_input_files({"h1": "a", "h2": "b", "h3": "ā"})
    lines = [json.loads(line) for f in files for line in f.splitlines()]
    assert [f.count(b"\n") for f in files] == [2, 1]
    assert lines[2] == {"custom_id": "h3", "body": {"input": ["ā"]}}


def test_ac_003_split_at_byte_cap(monkeypatch):
    monkeypatch.setattr(voyage_batch, "MAX_BYTES", 60)
    files = voyage_batch.build_input_files({"h1": "x" * 20, "h2": "y" * 20})
    assert len(files) == 2


# ── sync collection ───────────────────────────────────────────────────────────

@pytest.fixture
def sync_env(monkeypatch):
    calls: dict[str, list] = {"reindex": [], "registered": [], "deleted": [], "cached": [], "cleared": []}
    monkeypatch.setattr(sync, "get_latest_sha", lambda owner, repo, token: "b" * 40)
    monkeypatch.setattr(sync, "get_pending_batches", lambda conn, ns: ["batch-1"])
    monkeypatch.setattr(sync, "_full_reindex", lambda url, ns, token, use_batch=True:
                        calls["reindex"].append(use_batch) or True)
    monkeypatch.setattr(sync, "register_namespace", lambda conn, ns, url, sha: calls["registered"].append(sha))
    monkeypatch.setattr(sync, "delete_pending_batches", lambda conn, ids: calls["deleted"].append(ids))
    monkeypatch.setattr(sync, "cache_embeddings", lambda conn, b, rows: calls["cached"].append((b, rows)))
    monkeypatch.setattr(sync, "delete_cached_embeddings", lambda conn, ids: calls["cleared"].append(ids))
    monkeypatch.setattr(voyage_batch, "iter_output_embeddings", lambda file_id: iter([("h1", [1.0])]))
    return calls


NS = {"namespace": "ns", "repo_url": "https://github.com/o/r", "last_commit_sha": "a" * 40}


def test_ac_004_skips_namespace_while_batch_in_progress(monkeypatch, sync_env):
    monkeypatch.setattr(voyage_batch, "get_batch", lambda b: {"id": b, "status": "in_progress"})
    sync.sync_namespace(FakeConn(), NS, None)
    assert sync_env["reindex"] == [] and sync_env["registered"] == [] and sync_env["cached"] == []


def test_ac_005_collects_completed_batch_then_reindexes_from_cache(monkeypatch, sync_env):
    monkeypatch.setattr(voyage_batch, "get_batch", lambda b: {"id": b, "status": "completed", "output_file_id": "out"})
    sync.sync_namespace(FakeConn(), NS, None)
    assert sync_env["cached"] == [("batch-1", [("h1", [1.0])])]
    assert sync_env["reindex"] == [False]
    assert sync_env["registered"] == ["b" * 40]
    assert sync_env["deleted"] == [["batch-1"]] and sync_env["cleared"] == [["batch-1"]]


def test_ac_006_failed_batch_is_dropped_and_sync_raises(monkeypatch, sync_env):
    monkeypatch.setattr(voyage_batch, "get_batch", lambda b: {"id": b, "status": "failed", "errors": "bad"})
    with pytest.raises(RuntimeError, match="batch-1"):
        sync.sync_namespace(FakeConn(), NS, None)
    assert sync_env["deleted"] == [["batch-1"]]
    assert sync_env["reindex"] == [] and sync_env["registered"] == []


def test_submitted_full_reindex_does_not_advance_sha(monkeypatch, sync_env):
    monkeypatch.setattr(sync, "get_pending_batches", lambda conn, ns: [])
    monkeypatch.setattr(sync, "get_changed_files", lambda *a: ([], [], True))
    monkeypatch.setattr(sync, "_full_reindex", lambda url, ns, token, use_batch=True: False)
    advanced: list[str] = []
    monkeypatch.setattr(sync, "update_namespace_sha", lambda conn, ns, sha: advanced.append(sha))
    sync.sync_namespace(FakeConn(), NS, None)
    assert advanced == []
