"""scripts/register.py — forcing a reindex also clears any pending Voyage batch state."""

import sys

import pytest

from scripts import register


class FakeConn:
    def commit(self) -> None: ...
    def close(self) -> None: ...


@pytest.fixture
def env(monkeypatch):
    calls: dict[str, list] = {"registered": [], "cleared_pending": [], "cleared_cache": []}
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(register, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(register, "setup_meta_table", lambda conn: None)
    monkeypatch.setattr(register, "register_namespace", lambda conn, ns, url, sha: calls["registered"].append(sha))
    monkeypatch.setattr(register, "delete_pending_batches", lambda conn, ids: calls["cleared_pending"].append(ids))
    monkeypatch.setattr(register, "delete_cached_embeddings", lambda conn, ids: calls["cleared_cache"].append(ids))
    return calls


def test_ac_021_clearing_sha_also_clears_a_pending_batch(monkeypatch, env):
    monkeypatch.setattr(register, "get_pending_batches", lambda conn, ns: ["batch-1"])
    monkeypatch.setattr(
        sys, "argv", ["register.py", "--repo", "https://github.com/o/r", "--namespace", "ns", "--sha", ""]
    )
    register.main()
    assert env["registered"] == [None]
    assert env["cleared_pending"] == [["batch-1"]]
    assert env["cleared_cache"] == [["batch-1"]]


def test_ac_021_clearing_sha_with_no_pending_batch_clears_nothing(monkeypatch, env):
    monkeypatch.setattr(register, "get_pending_batches", lambda conn, ns: [])
    monkeypatch.setattr(
        sys, "argv", ["register.py", "--repo", "https://github.com/o/r", "--namespace", "ns", "--sha", ""]
    )
    register.main()
    assert env["cleared_pending"] == [] and env["cleared_cache"] == []


def test_ac_021_registering_a_real_sha_does_not_touch_pending_batches(monkeypatch, env):
    monkeypatch.setattr(register, "get_pending_batches", lambda conn, ns: ["batch-1"])
    monkeypatch.setattr(
        sys, "argv", ["register.py", "--repo", "https://github.com/o/r", "--namespace", "ns", "--sha", "abc123"]
    )
    register.main()
    assert env["registered"] == ["abc123"]
    assert env["cleared_pending"] == [] and env["cleared_cache"] == []
