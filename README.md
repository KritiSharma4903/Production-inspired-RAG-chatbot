# Production RAG Chatbot — Reference Implementation

A modular-monolith RAG system built to demonstrate how **document ingestion,
incremental updates, metadata synchronization, and vector database updates**
actually work in production — not a LangChain tutorial demo.

Stack: **FastAPI** (backend) · **Streamlit** (frontend) · **Groq** (LLM) ·
**Pinecone** (vector DB) · **PostgreSQL** (metadata DB) · local file storage
(S3-swappable).

---

## Why a Modular Monolith (not microservices)?

Every "service" named in the brief (Parser, Chunker, Embedder, Retriever...)
is a **Python module with a single responsibility**, not a separately
deployed microservice. This is a deliberate production engineering choice:

- At the scale most RAG systems actually operate at (thousands to low
  millions of documents), the operational overhead of microservices
  (service discovery, distributed tracing, network failure handling between
  every step) costs more than it buys.
- A modular monolith still gives you the thing that actually matters —
  **clean interfaces between components** — so any single module (e.g. the
  embedder) can be extracted into its own deployed service later *without
  rewriting the module itself*, only its call site (see `enqueue_ingestion_job`
  in `app/workers/tasks.py` for exactly where that seam is).
- The one place true horizontal scaling matters immediately is the
  **background worker pool** (parsing/embedding large PDFs) — that's why
  ingestion is already designed to run async via a queue, decoupled from the
  API process.

---

## Project Structure — and why each folder exists

```
rag_chatbot/
├── app/
│   ├── main.py                # composition root: wires routers, startup checks
│   ├── config.py              # ONE typed settings object, fail-fast on missing env
│   ├── logging_config.py      # structured, per-module loggers
│   ├── db/
│   │   ├── models.py          # SQLAlchemy ORM models (mirrors sql/schema.sql)
│   │   └── session.py         # connection pooling, get_db() dependency
│   ├── storage/
│   │   └── file_storage.py    # Adapter pattern: local disk today, S3 tomorrow
│   ├── ingestion/
│   │   ├── parser.py          # raw file -> normalized TextBlocks
│   │   ├── chunker.py         # TextBlocks -> content-hashed Chunks
│   │   ├── hasher.py          # ALL hashing logic, centralized
│   │   ├── diff_engine.py     # THE core update algorithm (hash-set diff)
│   │   ├── embedder.py        # batched, retried embedding calls
│   │   └── pipeline.py        # orchestrator: ties every step together
│   ├── vectorstore/
│   │   └── pinecone_client.py # ALL Pinecone SDK calls, isolated
│   ├── retrieval/
│   │   ├── retriever.py       # question -> embedding -> ANN search
│   │   ├── prompt_builder.py  # retrieved chunks -> LLM prompt
│   │   └── llm_service.py     # Groq call, isolated
│   ├── api/
│   │   ├── routes_documents.py
│   │   ├── routes_chat.py
│   │   └── routes_health.py
│   └── workers/
│       └── tasks.py           # async job wrapper + retry/backoff
├── frontend/
│   └── streamlit_app.py       # thin UI, HTTP-only calls to the backend
├── sql/
│   └── schema.sql             # canonical schema (source of truth for DBAs)
└── docs/
    ├── ARCHITECTURE.md        # Parts 1, 3, 4 — full system + DB + vector design
    ├── UPDATE_PIPELINE.md     # Part 6 — the update workflow, sequence diagrams
    ├── COST_ANALYSIS.md       # Part 12 — numerical cost comparisons
    ├── SCALABILITY.md         # Part 11 — 100 docs -> 10M docs
    └── INTERVIEW_QUESTIONS.md # Part 13
```

**Why this separation?** Production teams split code this way so that:
1. A bug or change in "how we chunk text" can never accidentally break
   "how we call Pinecone" — each module has exactly one reason to change
   (Single Responsibility Principle).
2. Any module can be unit-tested with mocked inputs/outputs, without
   spinning up Postgres, Pinecone, or Groq.
3. Junior engineers can be handed ONE folder to own without needing to
   understand the whole system.

---

## Setup

```bash
cd rag_chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in DATABASE_URL, PINECONE_API_KEY, GROQ_API_KEY

# create the Postgres database, then apply the schema
psql $DATABASE_URL -f sql/schema.sql

# start the backend
uvicorn app.main:app --reload --port 8000

# in a second terminal, start the frontend
streamlit run frontend/streamlit_app.py
```

## Try the update workflow yourself

1. Upload `handbook_v1.pdf` via the Streamlit UI.
2. Check `GET /documents/{document_id}/status` — note `total_chunks`.
3. Edit ONE paragraph in the PDF, save as a new file but **keep the same
   filename on upload** (the pipeline matches on filename → treats it as an
   update, not a new document).
4. Upload again. Check the API response: `chunks_embedded` will be ~1,
   `chunks_unchanged` will be ~everything else. That's the whole point —
   see `docs/UPDATE_PIPELINE.md` for exactly why.

See `docs/` for the deep-dive explanations requested in the brief (Parts 1,
3, 4, 6, 11, 12, 13). Every source file also carries a module-level
docstring explaining *why it exists*, not just what it does — read those
inline as you go.
