"""Retrieval evaluator (scripts/eval_retrieval.py) and rag.query.search — DB and Voyage are faked."""

from pathlib import Path

import pytest

from rag import query
from scripts import eval_retrieval as ev

REPO_ROOT = Path(__file__).parent.parent


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "q.toml"
    path.write_text(body, encoding="utf-8")
    return path


# ── Question loading ──────────────────────────────────────────────────────────

def test_load_questions_defaults_id_and_normalises_paths(tmp_path):
    path = _write(tmp_path, """
[[questions]]
question = "  Where is X?  "
files = ["./rag/db.py", "scripts\\\\sync.py", ".github/"]
""")
    [q] = ev.load_questions(path)
    assert q == ev.Question("1", "Where is X?", ("rag/db.py", "scripts/sync.py", ".github/"))


@pytest.mark.parametrize("body, message", [
    ("", "no \\[\\[questions\\]\\]"),
    ('[[questions]]\nquestion = "q"\n', "non-empty 'files'"),
    ('[[questions]]\nquestion = "q"\nfiles = []\n', "non-empty 'files'"),
    ('[[questions]]\nquestion = " "\nfiles = ["a"]\n', "no 'question' text"),
    ('questions = ["q"]\n', "must be an array of \\[\\[questions\\]\\] tables"),
    ('[[questions]]\nquestion = "q"\nfiles = ["./"]\n', "non-empty 'files'"),
    ('[[questions]]\nid = "a"\nquestion = "q"\nfiles = ["x"]\n'
     '[[questions]]\nid = "a"\nquestion = "r"\nfiles = ["y"]\n', "duplicate question ids"),
])
def test_load_questions_rejects_malformed_files(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        ev.load_questions(_write(tmp_path, body))


def test_bundled_question_set_points_at_real_files():
    questions = ev.load_questions(REPO_ROOT / "evals" / "github-repo-rag.toml")
    missing = [(q.id, f) for q in questions for f in q.files if not (REPO_ROOT / f).exists()]
    assert missing == []


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_matches_exact_path_or_directory_prefix():
    assert ev.matches("rag/db.py", "rag/db.py")
    assert not ev.matches("rag/db.py", "rag/db.pyc")
    assert ev.matches("rag/", "rag/query.py")
    assert not ev.matches("rag", "rag/query.py")


def test_recall_counts_expected_entries_found():
    assert ev.recall(("a.py", "b.py"), ["b.py", "c.py"]) == 0.5
    assert ev.recall(("pkg/",), ["pkg/x.py", "pkg/y.py"]) == 1.0
    assert ev.recall(("a.py",), []) == 0.0


def test_reciprocal_rank_uses_first_matching_file():
    assert ev.reciprocal_rank(("b.py", "c.py"), ["a.py", "c.py", "b.py"]) == 0.5
    assert ev.reciprocal_rank(("z.py",), ["a.py"]) == 0.0


def test_distinct_files_keeps_first_occurrence_order():
    assert ev.distinct_files(["b.py", "a.py", "b.py", "c.py"]) == ["b.py", "a.py", "c.py"]


def test_files_at_takes_top_k_chunks_per_table_in_table_order():
    o = ev.Outcome("1", "ns", "hybrid", {
        "ns_code": ["a.py", "a.py", "b.py", "c.py"],
        "ns_docs": ["README.md", "a.py", "spec.md"],
    })
    assert o.files_at(2) == ["a.py", "README.md"]
    assert o.files_at(3) == ["a.py", "b.py", "README.md", "spec.md"]
    assert o.files_at(4) == ["a.py", "b.py", "c.py", "README.md", "spec.md"]


def test_best_reciprocal_rank_ranks_each_table_separately():
    o = ev.Outcome("1", "ns", "hybrid", {
        "ns_code": [f"src/{i}.py" for i in range(20)],
        "ns_docs": ["README.md"],
    })
    # README.md is first in its own table, not behind 20 code files
    assert o.best_reciprocal_rank(("README.md",)) == 1.0
    assert o.best_reciprocal_rank(("src/2.py", "missing.md")) == pytest.approx(1 / 3)
    assert o.best_reciprocal_rank(("missing.md",)) == 0.0


def test_summarise_aggregates_hit_recall_mrr_and_latency():
    questions = [ev.Question("1", "q1", ("a.py", "b.py")), ev.Question("2", "q2", ("z.py",))]
    outcomes = [
        ev.Outcome("1", "ns", "hybrid", {"ns_code": ["x.py", "a.py", "b.py"]}, 10.0),
        ev.Outcome("2", "ns", "hybrid", {"ns_code": ["x.py"]}, 30.0),
    ]
    s = ev.summarise(outcomes, questions, [2, 5])
    assert s["hit@2"] == 0.5 and s["recall@2"] == 0.25
    assert s["hit@5"] == 0.5 and s["recall@5"] == 0.5
    assert s["mrr"] == 0.25  # (1/2 + 0) / 2
    assert s["search_ms_p50"] == 20.0


def test_disagreements_lists_only_questions_where_namespaces_differ():
    questions = [ev.Question("1", "q1", ("a.py",)), ev.Question("2", "q2", ("b.py",))]
    outcomes = [
        ev.Outcome("1", "small_chunks", "vector", {"t": ["a.py"]}),
        ev.Outcome("1", "large_chunks", "vector", {"t": ["x.py"]}),
        ev.Outcome("2", "small_chunks", "vector", {"t": ["b.py"]}),
        ev.Outcome("2", "large_chunks", "vector", {"t": ["b.py"]}),
    ]
    assert ev.disagreements(outcomes, questions, 6) == [
        {"mode": "vector", "id": "1", "question": "q1", "hit": {"small_chunks": True, "large_chunks": False}},
    ]


# ── evaluate() ────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_search(monkeypatch):
    calls: list[tuple] = []

    def search(conn, question, tables, top_k, query_embedding=None, mode="hybrid"):
        [table] = tables
        calls.append((question, table, mode, top_k, query_embedding is not None))
        return [("text", f"{table}/{mode}/{i}.py", "") for i in range(top_k)]

    monkeypatch.setattr(ev, "search", search)
    monkeypatch.setattr(ev, "namespace_tables", lambda conn, ns: [f"{ns}_code", f"{ns}_docs"])
    return calls


def test_evaluate_embeds_once_per_question_and_searches_each_table_once_at_max_k(fake_search):
    embedded: list[tuple[str, str]] = []
    questions = [ev.Question("1", "q1", ("x",)), ev.Question("2", "q2", ("y",))]

    outcomes, embed_ms = ev.evaluate(
        None, "ns", questions, ["hybrid", "vector", "fts"], [6, 20],
        embed_fn=lambda text, input_type: embedded.append((text, input_type)) or [0.1, 0.2],
    )

    assert embedded == [("q1", "query"), ("q2", "query")]
    assert len(embed_ms) == 2
    assert len(fake_search) == 2 * 3 * 2  # questions × modes × tables
    assert {top_k for *_, top_k, _ in fake_search} == {20}
    assert len(outcomes) == 2 * 3
    first = outcomes[0]
    assert list(first.table_files) == ["ns_code", "ns_docs"]
    assert len(first.files_at(6)) == 12 and len(first.files_at(20)) == 40


def test_evaluate_fts_only_never_embeds(fake_search):
    def no_embed(text, input_type):
        raise AssertionError("fts mode must not embed")

    outcomes, embed_ms = ev.evaluate(None, "ns", [ev.Question("1", "q", ("a",))], ["fts"], [6], embed_fn=no_embed)
    assert embed_ms == [] and len(outcomes) == 1
    assert fake_search == [("q", "ns_code", "fts", 6, False), ("q", "ns_docs", "fts", 6, False)]


# ── rag.query.search ──────────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, conn) -> None:
        self.conn = conn

    def execute(self, sql, params) -> None:
        if "to_regclass" in sql:
            self.rows = [(t,) for t in params[0] if t not in self.conn.missing]
            return
        table = sql.split("FROM ")[1].split()[0]
        self.conn.executed.append(table)
        self.rows = [("text", f"{table}.py", "")]

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _Conn:
    def __init__(self, missing=()) -> None:
        self.missing = set(missing)
        self.executed: list[str] = []

    def cursor(self):
        return _Cursor(self)


def test_search_needs_an_embedding_outside_fts_mode():
    with pytest.raises(ValueError, match="needs a query_embedding"):
        query.search(_Conn(), "q", ["t"], 6, mode="vector")


def test_search_never_queries_a_missing_table():
    conn = _Conn(missing={"ns_code"})
    rows = query.search(conn, "q", ["ns_code", "ns_docs"], 6, mode="fts")
    assert rows == [("text", "ns_docs.py", "")]
    assert conn.executed == ["ns_docs"]


def test_search_keeps_table_order():
    rows = query.search(_Conn(), "q", ["ns_code", "ns_docs"], 6, query_embedding=[0.0], mode="vector")
    assert [r[1] for r in rows] == ["ns_code.py", "ns_docs.py"]
