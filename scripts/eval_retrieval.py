#!/usr/bin/env python3
"""
Retrieval evaluation: score how well namespaces find the files that answer a fixed question set.

Each question lists the file paths that answer it. For every namespace and search mode
the script runs the same SQL as search_codebase and reports, at each cut-off k:

    hit@k     share of questions where at least one expected file is retrieved
    recall@k  mean share of a question's expected files that are retrieved
    MRR       mean reciprocal rank of the first expected file (at the largest k)

k is per table, as in search_codebase: a k of 6 returns up to 6 code and 6 doc chunks.
Modes: hybrid (what search_codebase returns), vector (embedding only) and fts (full-text
only, the no-embedding baseline).

Index the same repo under several namespaces (one per embedding model) and pass them all
to compare models on identical questions.

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
    DATABASE_URL    — PostgreSQL connection string
    VOYAGE_API_KEY  — query embedding (hybrid and vector modes)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from rag.db import get_conn, sanitise_namespace
from rag.query import SEARCH_MODES, embed, search

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
    ranked_files: dict[int, list[str]]  # k → distinct file paths in result order
    search_ms: list[float] = field(default_factory=list)


# ── Questions ─────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[Question]:
    """Parse and validate a question file; raises ValueError on any malformed entry."""
    with path.open("rb") as f:
        raw = tomllib.load(f).get("questions")
    if not raw:
        raise ValueError(f"{path}: no [[questions]] entries")

    questions: list[Question] = []
    for i, entry in enumerate(raw, start=1):
        qid = str(entry.get("id", i))
        text = entry.get("question", "")
        files = entry.get("files", [])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: question {qid} has no 'question' text")
        if not isinstance(files, list) or not files or not all(isinstance(p, str) and p for p in files):
            raise ValueError(f"{path}: question {qid} needs a non-empty 'files' list of paths")
        questions.append(Question(qid, text.strip(), tuple(_normalise(p) for p in files)))

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


def distinct_files(rows: list[tuple[str, str, str]]) -> list[str]:
    """File paths from search rows, first occurrence order, duplicates dropped."""
    seen: dict[str, None] = {}
    for _, file_path, _ in rows:
        seen.setdefault(file_path, None)
    return list(seen)


def summarise(outcomes: list[Outcome], questions: list[Question], ks: list[int]) -> dict:
    """Aggregate hit@k, recall@k and MRR (at the largest k) over the questions."""
    by_id = {q.id: q for q in questions}
    n = len(outcomes)
    summary: dict = {}
    for k in ks:
        recalls = [recall(by_id[o.question_id].files, o.ranked_files[k]) for o in outcomes]
        summary[f"hit@{k}"] = sum(r > 0 for r in recalls) / n
        summary[f"recall@{k}"] = sum(recalls) / n
    top = max(ks)
    summary["mrr"] = sum(reciprocal_rank(by_id[o.question_id].files, o.ranked_files[top]) for o in outcomes) / n
    times = [ms for o in outcomes for ms in o.search_ms]
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

def namespace_tables(conn, namespace: str) -> list[str]:
    """The namespace's _code and _docs tables that exist; raises if neither does."""
    tables = [f"{namespace}_code", f"{namespace}_docs"]
    with conn.cursor() as cur:
        cur.execute("SELECT t FROM unnest(%s::text[]) t WHERE to_regclass(t) IS NOT NULL", (tables,))
        found = {r[0] for r in cur.fetchall()}
    if not found:
        raise SystemExit(f"Namespace '{namespace}' has no {tables[0]} or {tables[1]} table — ingest it first.")
    return [t for t in tables if t in found]


def evaluate(
    conn,
    namespace: str,
    questions: list[Question],
    modes: list[str],
    ks: list[int],
    embed_fn=embed,
) -> tuple[list[Outcome], list[float]]:
    """Run every question through every mode. Returns the outcomes and per-question embed times (ms)."""
    from pgvector import Vector

    tables = namespace_tables(conn, namespace)
    needs_embedding = any(m != "fts" for m in modes)
    outcomes: list[Outcome] = []
    embed_ms: list[float] = []

    for q in questions:
        query_embedding = None
        if needs_embedding:
            start = time.perf_counter()
            query_embedding = Vector(embed_fn(q.question, input_type="query"))
            embed_ms.append((time.perf_counter() - start) * 1000)

        for mode in modes:
            outcome = Outcome(q.id, namespace, mode, ranked_files={})
            for k in ks:
                start = time.perf_counter()
                rows = search(conn, q.question, tables, k, query_embedding, mode=mode)
                outcome.search_ms.append((time.perf_counter() - start) * 1000)
                outcome.ranked_files[k] = distinct_files(rows)
            outcomes.append(outcome)

    return outcomes, embed_ms


def disagreements(outcomes: list[Outcome], questions: list[Question], k: int) -> list[dict]:
    """Questions where namespaces differ on whether an expected file is in the top k, per mode."""
    by_id = {q.id: q for q in questions}
    grouped: dict[tuple[str, str], dict[str, bool]] = {}
    for o in outcomes:
        hit = recall(by_id[o.question_id].files, o.ranked_files[k]) > 0
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
    modes: list[str],
    ks: list[int],
    diffs: list[dict],
    n_questions: int,
) -> None:
    print(f"\n{n_questions} questions · k is per table (code + docs)\n")
    metric_cols = [c for k in ks for c in (f"hit@{k}", f"recall@{k}")] + ["mrr"]
    width = max(len(ns) for ns in namespaces) + 2
    for mode in modes:
        print(f"── {mode} ".ljust(60, "─"))
        print("namespace".ljust(width) + "".join(c.rjust(10) for c in metric_cols) + "search ms p50/p95".rjust(20))
        for ns in namespaces:
            s = summaries[(ns, mode)]
            latency = f"{_fmt(s['search_ms_p50'])}/{_fmt(s['search_ms_p95'])}"
            cells = "".join(_fmt(s[c], ".3f" if c == "mrr" else ".0%").rjust(10) for c in metric_cols)
            print(ns.ljust(width) + cells + latency.rjust(20))
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
    namespaces = [sanitise_namespace(ns) for ns in args.namespaces]
    questions = load_questions(args.questions)

    conn = get_conn(os.environ["DATABASE_URL"])
    try:
        outcomes: list[Outcome] = []
        embed_times: dict[str, list[float]] = {}
        for ns in namespaces:
            print(f"Evaluating {ns} ({len(questions)} questions × {len(args.modes)} modes)...", flush=True)
            ns_outcomes, embed_times[ns] = evaluate(conn, ns, questions, args.modes, ks)
            outcomes.extend(ns_outcomes)
    finally:
        conn.close()

    summaries = {
        (ns, mode): summarise([o for o in outcomes if o.namespace == ns and o.mode == mode], questions, ks)
        for ns in namespaces for mode in args.modes
    }
    diffs = disagreements(outcomes, questions, min(ks))
    print_report(summaries, embed_times, namespaces, args.modes, ks, diffs, len(questions))

    if args.out:
        by_id = {q.id: q for q in questions}
        args.out.write_text(json.dumps({
            "questions_file": str(args.questions),
            "ks": ks,
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
                    "retrieved": {str(k): files for k, files in o.ranked_files.items()},
                    "recall": {
                        str(k): recall(by_id[o.question_id].files, files) for k, files in o.ranked_files.items()
                    },
                }
                for o in outcomes
            ],
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
