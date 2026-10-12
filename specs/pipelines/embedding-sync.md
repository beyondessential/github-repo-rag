# Pipeline Spec: `embedding-sync`

## Identity

| Field | Value |
|---|---|
| **Name** | Embedding reuse, Voyage Batch API full reindexes, and index pruning |
| **Type** | Scheduled script (`scripts/sync.py`) + CLIs (`scripts/ingest.py`, `scripts/reindex.py`) |
| **Status** | `implemented` |
| **Owner** | `@julianam-w` |
| **Linear issue** | N/A |
| **Repo** | `github-repo-rag` |
| **Trigger** | Railway cron (sync) / manual (ingest, reindex) |
| **Schedule** | `0 * * * *` (hourly, Railway cron service) |
| **Created** | 2026-10-11 |
| **Last updated** | 2026-10-12 |

## Purpose

Keep each namespace's index in step with its repo while minimising Voyage spend. Text that already has an embedding is never re-embedded. Large full reindexes go through the Voyage Batch API, which costs 33% less than real-time `/v1/embeddings` but can take up to 12 hours. Incremental syncs stay real time so the index stays fresh. Full reindexes also remove chunks the repo no longer produces.

## Inputs

- **External sources:** Voyage `/v1/embeddings`, Files API (`/v1/files`) and Batch API (`/v1/batches`) over HTTPS; GitHub clone and compare API.
- **Required configuration:** `VOYAGE_API_KEY`, `DATABASE_URL`; optional `VOYAGE_BATCH_THRESHOLD`, `GITHUB_TOKEN`.

## Outputs

- **Tables:** `{namespace}_code` / `{namespace}_docs` (now with `content_hash`), `rag_pending_batches` (one row per in-flight batch), `rag_embedding_cache` (batch results awaiting upsert).
- **Downstream consumers:** MCP server search (reads the namespace tables only).

## Processing logic

- **BL-001:** Only full reindexes (no baseline, diff too large, or `ingest.py --batch`) are eligible for the Batch API; incremental syncs always embed in real time.
- **BL-002:** A full reindex uses the Batch API when the number of distinct texts still needing an embedding, after reuse from the namespace tables and the cache, is at least `VOYAGE_BATCH_THRESHOLD` (default 2000).
- **BL-003:** Each batch input line embeds one text with `custom_id` set to its `content_hash`, using model `voyage-code-3` and `input_type` `document`.
- **BL-004:** A batch holds at most 100,000 lines and 100 MB of input; more pending texts are split across several batches.
- **BL-005:** Submitting a batch records it in `rag_pending_batches` and leaves indexed chunks unchanged; a submitting sync leaves `last_commit_sha` unchanged, while `ingest.py --batch` registers the namespace with no SHA so sync collects it.
- **BL-006:** While a namespace has an in-flight batch, sync skips that namespace.
- **BL-007:** When every batch for a namespace has reached a terminal status, sync loads each successful output line into `rag_embedding_cache`, then runs a real-time full reindex that reads embeddings from the cache before calling Voyage.
- **BL-008:** After a successful collecting reindex, sync advances `last_commit_sha`, deletes the namespace's `rag_pending_batches` rows and deletes the cache rows loaded from those batches.
- **BL-009:** A batch that ends `failed` or `cancelled` makes the namespace's sync fail and deletes all of that namespace's pending batch rows while keeping cached results, so the next run resubmits only uncached texts.
- **BL-010:** Cache rows older than 7 days are deleted at the start of each sync.
- **BL-011:** Every stored chunk carries a SHA-256 `content_hash` of its text, and embedding reuses a stored embedding with the same hash in the target table instead of calling Voyage.
- **BL-012:** Identical texts within one run are embedded once.
- **BL-013:** An incremental sync or reindex embeds every changed chunk before deleting anything, then deletes stale chunks and upserts new ones in a single transaction.
- **BL-014:** Once every upsert of a full reindex has succeeded, chunks whose ids the checkout no longer produces are deleted.
- **BL-015:** A full reindex of a checkout with no indexable files fails without changing the index.
- **BL-016:** A table that produces zero chunks in a run is left untouched rather than pruned; an empty keep set is never read as "keep nothing".
- **BL-017:** A malformed `VOYAGE_BATCH_THRESHOLD` fails only the namespace attempting a full reindex, not the whole sync run.
- **BL-018:** The full reindex that collects a completed batch advances `last_commit_sha` and clears batch state only once that reindex has itself completed.
- **BL-019:** A table's `content_hash` column, backfill and index are created only while the column is missing; once present, a later `setup_table` call does not repeat the migration.
- **BL-020:** `ingest()` closes its database connection on every exit path, including an unhandled error partway through embedding, batch submission or pruning.
- **BL-021:** Clearing a namespace's SHA to force a full reindex also clears any pending Voyage batch and cached embeddings recorded for it.

## SLAs

| SLA | Target |
|---|---|
| Freshness (incremental) | Within one hour |
| Freshness (full reindex via batch) | Within one hour of the batch completing (≤ 13 h after submission) |

## Error handling

| Failure mode | Behaviour | Alerting |
|---|---|---|
| Voyage real-time error during incremental sync | Transaction never opened; index unchanged; SHA not advanced (BL-013) | Cron job exit code |
| Voyage upload / create error | Raise; namespace sync fails | Cron job exit code |
| Batch `failed` / `cancelled` | BL-009 | Cron job exit code |
| Batch completed with some failed requests | Missing texts embed in real time during collection | Logged count |
| Empty or broken checkout | BL-015 | Cron job exit code |
| One table produces zero chunks this run | BL-016 | Logged, not an error |
| Malformed `VOYAGE_BATCH_THRESHOLD` | Raises `ValueError`; only that namespace's sync fails (BL-017) | Logged per namespace, cron job continues |
| Collecting reindex does not complete | Batch state and `last_commit_sha` left unchanged for retry (BL-018) | Logged, cron job continues |
| Namespace stuck on an unresolvable pending batch | `register.py --sha ""` clears it (BL-021) | Manual operator action |

## Acceptance criteria

| ID | Criterion | Implements |
|---|---|---|
| AC-001 | A full reindex below the threshold embeds in real time and submits no batch | BL-002 |
| AC-002 | A full reindex at or above the threshold submits batches, upserts nothing, and `ingest.py --batch` registers no SHA | BL-002, BL-005 |
| AC-003 | Batch input lines carry `custom_id` = content hash and one input each, split at the line/byte cap | BL-003, BL-004 |
| AC-004 | Sync skips a namespace whose batch is in progress | BL-006 |
| AC-005 | Sync loads completed output into the cache, reindexes with batching disabled, then clears pending rows and cache | BL-007, BL-008 |
| AC-006 | A failed batch deletes every pending row for the namespace and the sync raises | BL-009 |
| AC-007 | Batch creation requests model `voyage-code-3` and `input_type` `document` | BL-003 |
| AC-008 | Only successful output lines are loaded into the cache | BL-007 |
| AC-009 | Embedding reuses stored or cached embeddings by hash and sends each new distinct text to Voyage once | BL-007, BL-011, BL-012 |
| AC-010 | The SQL backfill hash equals the Python `content_hash` | BL-011 |
| AC-011 | Incremental sync embeds before deleting, commits once, and leaves the index untouched if embedding fails | BL-013 |
| AC-012 | A full reindex prunes stale chunks only after every upsert, and a batch submission prunes nothing | BL-014 |
| AC-013 | A full reindex of an empty checkout raises before touching the index | BL-015 |
| AC-014 | `delete_chunks_except` returns 0 and deletes nothing when `keep_ids` is empty | BL-016 |
| AC-015 | A full reindex with one table at zero chunks this run skips pruning that table | BL-016 |
| AC-016 | A malformed `VOYAGE_BATCH_THRESHOLD` raises `ValueError`, caught per namespace by sync's loop | BL-017 |
| AC-017 | The collecting reindex only advances `last_commit_sha` and clears batch state on its own success | BL-018 |
| AC-018 | `setup_table`'s migration block runs only while `content_hash` is missing from the table | BL-019 |
| AC-019 | `ingest()` closes its connection even when an embedding call raises | BL-020 |
| AC-020 | `register.py --sha ""` clears the namespace's pending batch rows and cached embeddings; a real SHA leaves them untouched | BL-021 |

## Open questions

| ID | Question | Owner | Due |
|---|---|---|---|
| OQ-001 | Voyage does not document a max upload file size; 100 MB per batch is a conservative assumption | @julianam-w | before merge |

## Change log

| Date | Author | Change |
|---|---|---|
| 2026-10-12 | @julianam-w | Implemented: embedding reuse, Voyage Batch API full reindexes, index pruning, hourly sync. All ACs pass by automated test; AC-001, AC-002 and AC-005 also confirmed live against the real Voyage API and database |
| 2026-10-12 | @julianam-w | Second `/code-review --fix` pass (8-angle): add BL-019..021 / AC-018..020 for a repeated-lock migration guard, a connection leak on error, and a `register.py` recovery path for a stuck pending batch |
