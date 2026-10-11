# Pipeline Spec: `batch-embedding`

## Identity

| Field | Value |
|---|---|
| **Name** | Voyage Batch API embedding for full reindexes |
| **Type** | Scheduled script (`scripts/sync.py`) + CLI (`scripts/ingest.py --batch`) |
| **Status** | `draft` |
| **Owner** | `@julianam-w` |
| **Linear issue** | N/A |
| **Repo** | `github-repo-rag` |
| **Trigger** | Railway cron (sync) / manual (ingest) |
| **Schedule** | `0 * * * *` (hourly, Railway cron service) |
| **Created** | 2026-10-11 |
| **Last updated** | 2026-10-11 |

## Purpose

Cut Voyage spend on large embedding runs by sending them through the Voyage Batch API, which is priced 33% below real-time `/v1/embeddings` in exchange for an up-to-12-hour completion window. Small incremental syncs stay real time so the index stays fresh.

## Inputs

- **External sources:** Voyage Files API (`/v1/files`) and Batch API (`/v1/batches`) over HTTPS; GitHub clone as today.
- **Required configuration:** `VOYAGE_API_KEY`, `DATABASE_URL`; optional `VOYAGE_BATCH_THRESHOLD`.

## Outputs

- **Tables:** `rag_pending_batches` (one row per in-flight batch), `rag_embedding_cache` (batch results awaiting upsert); `{namespace}_code` / `{namespace}_docs` as today.
- **Downstream consumers:** MCP server search (reads the namespace tables only).

## Processing logic

- **BL-001:** Only full reindexes (no baseline, diff too large, or `ingest.py --batch`) are eligible for the Batch API; incremental syncs always embed in real time.
- **BL-002:** A full reindex uses the Batch API when the number of distinct texts still needing an embedding, after reuse from the namespace tables and the cache, is at least `VOYAGE_BATCH_THRESHOLD` (default 2000).
- **BL-003:** Each batch input line embeds one text with `custom_id` set to its `content_hash`, using model `voyage-code-3` and `input_type` `document`.
- **BL-004:** A batch holds at most 100,000 lines and 100 MB of input; more pending texts are split across several batches.
- **BL-005:** Submitting a batch records it in `rag_pending_batches` and leaves the namespace's `last_commit_sha` and indexed chunks unchanged.
- **BL-006:** While a namespace has an in-flight batch, sync skips that namespace.
- **BL-007:** When every batch for a namespace has reached a terminal status, sync loads each successful output line into `rag_embedding_cache`, then runs a real-time full reindex that reads embeddings from the cache before calling Voyage.
- **BL-008:** After a successful collecting reindex, sync advances `last_commit_sha`, deletes the namespace's `rag_pending_batches` rows and deletes the cache rows loaded from those batches.
- **BL-009:** A batch that ends `failed` or `cancelled` has its row deleted and makes the sync for that namespace fail; the next run resubmits only the texts not already cached.
- **BL-010:** Cache rows older than 7 days are deleted at the start of each sync.

## SLAs

| SLA | Target |
|---|---|
| Freshness (incremental) | Within one hour |
| Freshness (full reindex via batch) | Within one hour of the batch completing (≤ 13 h after submission) |

## Error handling

| Failure mode | Behaviour | Alerting |
|---|---|---|
| Voyage upload / create error | Raise; namespace sync fails; nothing recorded | Cron job exit code |
| Batch `failed` / `cancelled` | BL-009 | Cron job exit code |
| Batch completed with some failed requests | Missing texts embed in real time during collection | Logged count |

## Acceptance criteria

| ID | Criterion | Implements |
|---|---|---|
| AC-001 | A full reindex below the threshold embeds in real time and submits no batch | BL-002 |
| AC-002 | A full reindex at or above the threshold submits batches and upserts nothing | BL-002, BL-005 |
| AC-003 | Batch input lines carry `custom_id` = content hash and one input each, split at the line/byte cap | BL-003, BL-004 |
| AC-004 | Sync skips a namespace whose batch is in progress | BL-006 |
| AC-005 | Sync loads completed output into the cache, reindexes with batching disabled, then clears pending rows and cache | BL-007, BL-008 |
| AC-006 | A failed batch row is deleted and the sync raises | BL-009 |

## Open questions

| ID | Question | Owner | Due |
|---|---|---|---|
| OQ-001 | Voyage does not document a max upload file size; 100 MB per batch is a conservative assumption | @julianam-w | before merge |

## Change log

| Date | Author | Change |
|---|---|---|
| 2026-10-11 | @julianam-w | Initial draft |
