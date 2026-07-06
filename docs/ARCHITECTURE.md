# Architecture (Parts 1, 3, 4)

## Part 1 — System Architecture

### Components

| Component | Module | Responsibility |
|---|---|---|
| Frontend | `frontend/streamlit_app.py` | Thin UI, HTTP calls only |
| API Layer | `app/api/routes_*.py` | Validation, auth (add via middleware), request/response shaping |
| Parser | `app/ingestion/parser.py` | Raw file → normalized `TextBlock` list |
| Chunking Service | `app/ingestion/chunker.py` | `TextBlock` → content-hashed `Chunk` list |
| Hashing Service | `app/ingestion/hasher.py` | All SHA-256 logic, chunk_id derivation |
| Embedding Service | `app/ingestion/embedder.py` | Batched, retried calls to the embedding model |
| Retriever | `app/retrieval/retriever.py` | Question → query embedding → ANN search |
| Prompt Builder | `app/retrieval/prompt_builder.py` | Retrieved chunks → LLM prompt |
| LLM Service | `app/retrieval/llm_service.py` | Groq API call |
| Vector Database | `app/vectorstore/pinecone_client.py` | Pinecone upsert/delete/query, isolated |
| Metadata Database | `app/db/*` + `sql/schema.sql` | Source of truth for doc/chunk state |
| File Storage | `app/storage/file_storage.py` | Raw file persistence, S3-swappable |
| Logging | `app/logging_config.py` | Structured, per-module logs |
| Configuration | `app/config.py` | One typed settings object |
| Error Handling | try/except at API boundary + `tenacity` retries at I/O boundaries | Fail fast on bad input, retry on transient errors |
| Monitoring | see "Monitoring" below | Latency, error rate, cost tracking |
| Background Workers | `app/workers/tasks.py` | Async ingestion, decoupled from API latency |

### Architecture Diagram

```
┌────────────────┐        HTTP         ┌──────────────────────┐
│  Streamlit UI   │ ──────────────────▶ │     FastAPI (API)      │
└────────────────┘                     └──────────┬───────────┘
                                                   │
                    ┌──────────────────────────────┼───────────────────────────┐
                    ▼                                                          ▼
          ┌───────────────────┐                                    ┌───────────────────┐
          │   File Storage      │                                    │   Metadata DB       │
          │ (local disk / S3)   │                                    │   (PostgreSQL)      │
          └─────────┬───────────┘                                    └─────────┬───────────┘
                    │ file_hash check (idempotency gate)                       │ read old chunk hashes
                    ▼                                                          │
          ┌───────────────────┐                                                │
          │  Worker Queue       │◄───────────── enqueue job ────────────────────┘    
          └─────────┬───────────┘
                    ▼
          ┌───────────────────┐
          │  Parser Service     │
          └─────────┬───────────┘
                    ▼
          ┌───────────────────┐
          │ Chunking Service    │──▶ content_hash per chunk (Hashing Service)
          └─────────┬───────────┘
                    ▼
          ┌───────────────────┐        old hashes (from Metadata DB)
          │   Diff Engine       │◄───────────────────────────────────
          └─────────┬───────────┘
       ┌────────────┼─────────────┬───────────────┐
       ▼            ▼             ▼               ▼
  unchanged      changed       deleted         inserted
   (skip)           │             │                │
                     ▼             ▼                ▼
          ┌───────────────────┐  ┌───────────────────┐
          │ Embedding Service   │  │   Vector Database │
          │ (batched calls)     │─▶│ upsert / delete   │
          └───────────────────┘  │  (Pinecone, HNSW)    │
                                  └─────────┬───────────┘
                                            ▼
                                  ┌───────────────────┐
                                  │   Metadata DB       │
                                  │ (hash table sync,   │
                                  │  version bump)      │
                                  └─────────┬───────────┘
                                            ▼
                                  ┌───────────────────┐
                                  │Logging + Monitoring│
                                  └───────────────────┘

--------------------------- CHAT PATH ---------------------------

┌────────────┐   question   ┌───────────────┐   embed   ┌───────────────┐
│ Streamlit  │─────────────▶│   Retriever  │──────────▶│ Embedding Service│
└────────────┘               └───────┬───────┘           └───────────────┘
                                      │ top-k ANN search
                                      ▼
                              ┌───────────────┐
                              │Vector Database│
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                              │Prompt Builder │
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                              │ LLM Service   │──▶ Groq API
                              │(Groq client)  │
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                              │ Metadata DB   │  (chat_history persisted)
                              └───────────────┘
```

### Monitoring (what's tracked, and where)

| Metric | Why it matters |
|---|---|
| Per-stage latency (parse/chunk/diff/embed/upsert) | Find the bottleneck stage as documents grow |
| Diff ratio (`changed / total_chunks`) | A sustained spike means either the parser is unstable (producing different text for identical PDFs) or the chunking config changed — both need investigation |
| Embedding tokens consumed / day | Direct line to cost — see `docs/COST_ANALYSIS.md` |
| Ingestion job failure rate | Health of parser/embedder/vector DB dependencies |
| Queue depth | Signals need to scale worker pool horizontally |
| Pinecone query latency (p50/p95/p99) | Retrieval-path user experience |

In this reference project these are emitted as structured log lines
(`app/logging_config.py`) that a log aggregator (ELK/Datadog/CloudWatch)
would parse into dashboards and alerts in a real deployment.

---

## Part 3 — Database Design (PostgreSQL)

See `sql/schema.sql` for full `CREATE TABLE` statements with inline
column-by-column commentary. Summary of *why* each table exists:

- **`documents`** — one row per logical document. Holds `latest_file_hash`
  (cheapest possible "did anything change" check) and `latest_version`.
- **`document_versions`** — append-only history, never deleted. This is
  what makes rollback possible without re-uploading a file.
- **`chunks`** — THE hash table that powers the diff algorithm. Every
  update reads this table's `content_hash` column before re-embedding
  anything.
- **`chat_sessions` / `chat_history`** — conversational memory;
  `retrieved_chunk_ids` on each assistant message is a debugging/audit
  trail ("why did it answer this way?").
- **`ingestion_jobs`** — keyed by `(document_id, file_hash)` with a UNIQUE
  constraint, the backbone of idempotency: resubmitting identical bytes
  while a job is in flight is a safe no-op.
- **`users`** — optional, only needed for per-tenant isolation.

**Why Postgres and not "just Pinecone metadata" for all of this?** Pinecone
metadata filters are optimized for *retrieval-time filtering* (fast,
approximate, limited query expressiveness). Postgres is needed for
*relational* questions the diff engine and the API need answered
precisely and transactionally: "give me every chunk_id + hash for this
document," "has this exact file already been processed," "what changed
between v3 and v4." Those are exactly the queries B-tree indexes and ACID
transactions are built for.

---

## Part 4 — Vector Database Design (Pinecone)

### How Pinecone stores vectors

Pinecone indexes vectors using an ANN structure (HNSW-family under the
hood for its serverless indexes) inside a **namespace** — a soft partition
within an index. Every vector has:
- an **id** (we use `chunk_id`, so it's already unique and deterministic)
- **values** (the embedding, 384-dim for `all-MiniLM-L6-v2`)
- **metadata** (small JSON blob, filterable at query time)

### Metadata schema used in this project

```json
{
  "document_id": "uuid",
  "chunk_id": "uuid:16hexchars",
  "page": 250,
  "version": 4,
  "content_hash": "sha256 hex",
  "filename": "employee_handbook.pdf",
  "text": "first ~2000 chars of the chunk"
}
```

| Field | Why it exists |
|---|---|
| `document_id` | Enables scoping retrieval to one document AND bulk-deleting an entire document's vectors via a metadata filter, without enumerating every chunk_id |
| `chunk_id` | Redundant with Pinecone's own vector id, but kept as metadata so retrieval results carry it without a second lookup |
| `page` | Needed for citations shown to the user |
| `version` | Lets you filter to "only current version" vectors if you ever soft-retain stale vectors instead of hard-deleting them (a rollback strategy trade-off) |
| `content_hash` | Lets an audit/repair job cross-check Pinecone state against Postgres state without re-computing hashes |
| `filename` | Human-readable, shown directly in citations |
| `text` | Avoids a second round-trip to Postgres just to render an answer; traded off against metadata payload size limits for very large chunks |

Pinecone is treated as a **derived, rebuildable index** in this design —
if it were ever wiped, replaying every row in the `chunks` table (which
still has `content_hash` and enough info to reconstruct the vector-worthy
text, assuming raw files are retained in file storage) would fully restore
it. This is why Postgres, not Pinecone, is the system of record.
