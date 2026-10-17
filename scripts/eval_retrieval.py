#!/usr/bin/env python3
"""
Retrieval evaluation: score how well namespaces find the files that answer a fixed question set.

Each question lists the file paths that answer it. For every namespace and search mode
the script runs the same SQL as search_codebase and reports, at each cut-off k:

    hit@k     share of questions where at least one expected file is retrieved
    recall@k  mean share of a question's expected files that are retrieved
    MRR       mean of 1 / the best rank an expected file reaches, ranking the code and docs
              tables separately so an answer in either table counts the same

k is per table, as in search_codebase: a k of 6 returns up to 6 code and 6 doc chunks.
Each table is searched once at the largest k and cut down for the smaller ones, so the
search latency is that of one search_codebase-style query at the largest k.
Modes: hybrid (what search_codebase returns), vector (embedding only) and fts (full-text
only, the no-embedding baseline).

Several namespaces are scored on identical questions. Each namespace is embedded and
queried with the model it is registered with (rag_namespaces.embedding_model —
voyage-code-3 if unregistered), so namespaces indexed with different embedding models
compare meaningfully. The questions themselves are shared across namespaces regardless
of model.

Usage:
    uv run python scripts/eval_retrieval.py --questions evals/github-repo-rag.toml \\
        --namespaces github_repo_rag
    uv run python scripts/eval_retrieval.py --questions evals/tupaia.toml \\
        --namespaces tupaia tupaia_qwen06 --k 6 20 --out eval-results.json

Question file (TOML):
    [[questions]]
    id = "sync-full-reindex"            # optional; defaults to the question's position
    question = "When does sync rebuild the whole index?"
    files = ["scripts/sync.py"]         # an entry ending in "/" matches any file under it

Environment variables:
    DATABASE_URL        — PostgreSQL connection string
    VOYAGE_API_KEY      — query embedding for a voyage-code-3 namespace
    EMBEDDING_BASE_URL  — query embedding for a namespace registered with an
                          OpenAI-compatible model (see rag/embeddings.py)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from rag.db import get_conn, resolve_model, sanitise_namespace, setup_meta_table
from rag.query import SEARCH_MODES, existing_tables, search

if TYPE_CHECKING:
    import psycopg2.extensions

    from rag.embeddings import EmbeddingModel

DEFAULT_KS = [6, 20]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    files: tuple[str, ...]


@dataclass
class Outcome:
    """One question's result for one namespace and mode."""
    question_id: str
    namespace: str
    mode: str
    table_files: dict[str, list[str]]  # table → each chunk's file path in rank order, at the largest k
    search_ms: float = 0.0

    def files_at(self, k: int) -> list[str]:
        """Distinct files in each table's top k chunks, tables in order — what search_codebase returns."""
        return distinct_files(path for paths in self.table_files.values() for path in paths[:k])

    def best_reciprocal_rank(self, expected: tuple[str, ...]) -> float:
        """Best reciprocal rank an expected file reaches within any one table's distinct files."""
        return max(
            (reciprocal_rank(expected, distinct_files(paths)) for paths in self.table_files.values()),
            default=0.0,
        )


# ── Questions ─────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[Question]:
    """Parse and validate a question file; raises ValueError on any malformed entry."""
    with path.open("rb") as f:
        raw = tomllib.load(f).get("questions")
    if not raw:
        raise ValueError(f"{path}: no [[questions]] entries")
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise ValueError(f"{path}: 'questions' must be an array of [[questions]] tables")

    questions: list[Question] = []
    for i, entry in enumerate(raw, start=1):
        qid = str(entry.get("id", i))
        text = entry.get("question", "")
        files = entry.get("files", [])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: question {qid} has no 'question' text")
        if not isinstance(files, list) or not all(isinstance(p, str) for p in files):
            raise ValueError(f"{path}: question {qid} needs a non-empty 'files' list of paths")
        normalised = tuple(_normalise(p) for p in files)
        if not normalised or not all(normalised):
            raise ValueError(f"{path}: question {qid} needs a non-empty 'files' list of paths")
        questions.append(Question(qid, text.strip(), normalised))

    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate question ids {duplicates}")
    return questions


def _normalise(path: str) -> str:
    """Match the repo-relative, forward-slash form ingest stores in file_path."""
    path = path.replace("\\", "/")
    return path[2:] if path.startswith("./") else path


def matches(expected: str, file_path: str) -> bool:
    """An expected entry ending in '/' is a directory prefix; anything else is an exact path."""
    if expected.endswith("/"):
        return file_path.startswith(expected)
    return file_path == expected


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall(expected: tuple[str, ...], files: list[str]) -> float:
    """Share of expected entries matched by at least one retrieved file."""
    return sum(any(matches(e, f) for f in files) for e in expected) / len(expected)


def reciprocal_rank(expected: tuple[str, ...], files: list[str]) -> float:
    """1 / position of the first retrieved file matching any expected entry; 0 if none."""
    for rank, f in enumerate(files, start=1):
        if any(matches(e, f) for e in expected):
            return 1.0 / rank
    return 0.0


def distinct_files(paths: Iterable[str]) -> list[str]:
    """File paths in first-occurrence order, duplicates dropped."""
    return list(dict.fromkeys(paths))


def summarise(outcomes: list[Outcome], questions: list[Question], ks: list[int]) -> dict:
    """Aggregate hit@k, recall@k, MRR and search latency over the questions."""
    by_id = {q.id: q for q in questions}
    n = len(outcomes)
    summary: dict = {}
    for k in ks:
        recalls = [recall(by_id[o.question_id].files, o.files_at(k)) for o in outcomes]
        summary[f"hit@{k}"] = sum(r > 0 for r in recalls) / n
        summary[f"recall@{k}"] = sum(recalls) / n
    summary["mrr"] = sum(o.best_reciprocal_rank(by_id[o.question_id].files) for o in outcomes) / n
    times = [o.search_ms for o in outcomes]
    summary["search_ms_p50"] = _percentile(times, 50)
    summary["search_ms_p95"] = _percentile(times, 95)
    return summary


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]


# ── Run ───────────────────────────────────────────────────────────────────────

def namespace_tables(conn: psycopg2.extensions.connection, namespace: str) -> list[str]:
    """The namespace's _code and _docs tables that exist; raises if neither does."""
    tables = [f"{namespace}_code", f"{namespace}_docs"]
    found = existing_tables(conn, tables)
    if not found:
        raise SystemExit(f"Namespace '{namespace}' has no {tables[0]} or {tables[1]} table — ingest it first.")
    return found


def evaluate(
    conn: psycopg2.extensions.connection,
    namespace: str,
    questions: list[Question],
    modes: list[str],
    ks: list[int],
    model: EmbeddingModel,
) -> tuple[list[Outcome], list[float]]:
    """Run every question through every mode. Returns the outcomes and per-question embed times (ms).

    Each table is searched once at the largest k; smaller ks are prefixes of that ranking.
    model embeds each question — it must be the model the namespace is indexed with
    (see rag.db.resolve_model), since any other model's vectors are not comparable to it.
    """
    from pgvector import Vector

    tables = namespace_tables(conn, namespace)
    top = max(ks)
    needs_embedding = any(m != "fts" for m in modes)
    outcomes: list[Outcome] = []
    embed_ms: list[float] = []

    for q in questions:
        query_embedding = None
        if needs_embedding:
            start = time.perf_counter()
            query_embedding = Vector(model.embed([q.question], "query")[0])
            embed_ms.append((time.perf_counter() - start) * 1000)

        for mode in modes:
            table_files: dict[str, list[str]] = {}
            start = time.perf_counter()
            for table in tables:
                rows = search(conn, q.question, [table], top, query_embedding, mode=mode)
                table_files[table] = [file_path for _, file_path, _ in rows]
            elapsed_ms = (time.perf_counter() - start) * 1000
            outcomes.append(Outcome(q.id, namespace, mode, table_files, elapsed_ms))

    return outcomes, embed_ms


def disagreements(outcomes: list[Outcome], questions: list[Question], k: int) -> list[dict]:
    """Questions where namespaces differ on whether an expected file is in the top k, per mode."""
    by_id = {q.id: q for q in questions}
    grouped: dict[tuple[str, str], dict[str, bool]] = {}
    for o in outcomes:
        hit = recall(by_id[o.question_id].files, o.files_at(k)) > 0
        grouped.setdefault((o.mode, o.question_id), {})[o.namespace] = hit
    return [
        {"mode": mode, "id": qid, "question": by_id[qid].question, "hit": hits}
        for (mode, qid), hits in grouped.items()
        if len(set(hits.values())) > 1
    ]


def _fmt(value: float | None, spec: str = ".0f") -> str:
    return "-" if value is None else format(value, spec)


def print_report(
    summaries: dict[tuple[str, str], dict],
    embed_times: dict[str, list[float]],
    namespaces: list[str],
    models: dict[str, str],
    modes: list[str],
    ks: list[int],
    diffs: list[dict],
    n_questions: int,
) -> None:
    print(f"\n{n_questions} questions · k is per table (code + docs) · search ms is one query at k={max(ks)}\n")
    labels = {ns: f"{ns} ({models[ns]})" for ns in namespaces}
    metric_cols = [c for k in ks for c in (f"hit@{k}", f"recall@{k}")] + ["mrr"]
    width = max(len(label) for label in labels.values()) + 2
    for mode in modes:
        print(f"── {mode} ".ljust(60, "─"))
        print("namespace".ljust(width) + "".join(c.rjust(10) for c in metric_cols) + "search ms p50/p95".rjust(20))
        for ns in namespaces:
            s = summaries[(ns, mode)]
            latency = f"{_fmt(s['search_ms_p50'])}/{_fmt(s['search_ms_p95'])}"
            cells = "".join(_fmt(s[c], ".3f" if c == "mrr" else ".0%").rjust(10) for c in metric_cols)
            print(labels[ns].ljust(width) + cells + latency.rjust(20))
        print()

    if any(embed_times.values()):
        print("query embedding ms p50/p95: " + ", ".join(
            f"{ns} {_fmt(_percentile(t, 50))}/{_fmt(_percentile(t, 95))}" for ns, t in embed_times.items() if t
        ))

    if len(namespaces) > 1:
        print(f"\nQuestions where namespaces disagree at hit@{min(ks)}: {len(diffs)}")
        for d in diffs:
            hits = ", ".join(f"{ns}={'hit' if h else 'miss'}" for ns, h in d["hit"].items())
            print(f"  [{d['mode']}] {d['id']}: {hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval quality of indexed namespaces against a question set")
    parser.add_argument("--questions", required=True, type=Path, help="TOML question file")
    parser.add_argument("--namespaces", required=True, nargs="+", help="Namespaces to evaluate")
    parser.add_argument("--modes", nargs="+", choices=SEARCH_MODES, default=list(SEARCH_MODES))
    parser.add_argument("--k", dest="ks", nargs="+", type=int, default=DEFAULT_KS,
                        help=f"Per-table result cut-offs (default {DEFAULT_KS}; search_codebase uses 6)")
    parser.add_argument("--out", type=Path, help="Write per-question results and summaries as JSON")
    args = parser.parse_args()

    if any(k < 1 for k in args.ks):
        parser.error("--k values must be positive")
    ks = sorted(set(args.ks))
    namespaces = list(dict.fromkeys(sanitise_namespace(ns) for ns in args.namespaces))
    questions = load_questions(args.questions)

    conn = get_conn(os.environ["DATABASE_URL"])
    try:
        setup_meta_table(conn)
        outcomes: list[Outcome] = []
        embed_times: dict[str, list[float]] = {}
        models: dict[str, str] = {}
        for ns in namespaces:
            model = resolve_model(conn, ns)
            models[ns] = model.name
            print(
                f"Evaluating {ns} ({model.name}, {len(questions)} questions × {len(args.modes)} modes)...",
                flush=True,
            )
            ns_outcomes, embed_times[ns] = evaluate(conn, ns, questions, args.modes, ks, model)
            outcomes.extend(ns_outcomes)
    finally:
        conn.close()

    summaries = {
        (ns, mode): summarise([o for o in outcomes if o.namespace == ns and o.mode == mode], questions, ks)
        for ns in namespaces for mode in args.modes
    }
    diffs = disagreements(outcomes, questions, min(ks))
    print_report(summaries, embed_times, namespaces, models, args.modes, ks, diffs, len(questions))

    if args.out:
        by_id = {q.id: q for q in questions}
        args.out.write_text(json.dumps({
            "questions_file": str(args.questions),
            "ks": ks,
            "models": models,
            "summaries": [{"namespace": ns, "mode": m, **s} for (ns, m), s in summaries.items()],
            "embed_ms": embed_times,
            "disagreements": diffs,
            "results": [
                {
                    "id": o.question_id,
                    "question": by_id[o.question_id].question,
                    "expected": list(by_id[o.question_id].files),
                    "namespace": o.namespace,
                    "mode": o.mode,
                    "retrieved": {str(k): o.files_at(k) for k in ks},
                    "recall": {str(k): recall(by_id[o.question_id].files, o.files_at(k)) for k in ks},
                    "reciprocal_rank": o.best_reciprocal_rank(by_id[o.question_id].files),
                    "chunks_by_table": o.table_files,
                }
                for o in outcomes
            ],
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
