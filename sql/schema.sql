CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS documents (
    document_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename           TEXT NOT NULL,
    owner_user_id      UUID REFERENCES users(user_id),

    latest_file_hash   TEXT NOT NULL,

    latest_version     INTEGER NOT NULL DEFAULT 1,

    parser_version     TEXT NOT NULL DEFAULT 'v1',

    total_pages        INTEGER,
    total_chunks       INTEGER NOT NULL DEFAULT 0,

    ingestion_status   TEXT NOT NULL DEFAULT 'uploaded',
    last_error         TEXT,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(ingestion_status);

CREATE TABLE IF NOT EXISTS document_versions (
    version_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version_number   INTEGER NOT NULL,
    file_hash        TEXT NOT NULL,
    storage_path     TEXT NOT NULL,     -- where the raw file for THIS version lives
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    changed_chunks   INTEGER NOT NULL DEFAULT 0,
    deleted_chunks    INTEGER NOT NULL DEFAULT 0,
    inserted_chunks  INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, version_number)
);


CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,   -- deterministic: see hasher.py
    document_id      UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    content_hash     TEXT NOT NULL,

    page_number      INTEGER,
    order_index      INTEGER NOT NULL,
    char_start       INTEGER,
    char_end         INTEGER,
    token_count      INTEGER,

    version_added    INTEGER NOT NULL,
    version_updated  INTEGER NOT NULL,

    vector_id        TEXT NOT NULL,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_version_updated ON chunks(document_id, version_updated);


CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID REFERENCES users(user_id),
    title          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_history (
    message_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,

    retrieved_chunk_ids TEXT[],

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id, created_at);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    file_hash       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued', -- queued|running|succeeded|failed
    attempts        INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, file_hash)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name     TEXT NOT NULL,
    dataset_name        TEXT NOT NULL,
    evaluator_used       TEXT NOT NULL,   -- 'ragas' | 'langsmith' | 'native' | 'deepeval'
    total_questions     INTEGER NOT NULL DEFAULT 0,

    avg_context_precision  DOUBLE PRECISION,
    avg_context_recall     DOUBLE PRECISION,
    avg_context_relevancy  DOUBLE PRECISION,
    avg_context_utilization DOUBLE PRECISION,
    avg_faithfulness       DOUBLE PRECISION,
    avg_answer_relevancy   DOUBLE PRECISION,
    avg_answer_correctness DOUBLE PRECISION,
    avg_semantic_similarity DOUBLE PRECISION,
    overall_score           DOUBLE PRECISION,  -- mean of all available metric averages

    pass_count          INTEGER NOT NULL DEFAULT 0,
    fail_count          INTEGER NOT NULL DEFAULT 0,
    execution_time_sec  DOUBLE PRECISION,

    status              TEXT NOT NULL DEFAULT 'running', -- running|completed|failed
    error_message        TEXT,

    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_experiment ON evaluation_runs(experiment_name, started_at);


CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id               UUID NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,

    question             TEXT NOT NULL,
    generated_answer     TEXT,
    ground_truth         TEXT,
    retrieved_contexts   TEXT[],

    context_precision    DOUBLE PRECISION,
    context_recall       DOUBLE PRECISION,
    context_relevancy    DOUBLE PRECISION,
    context_utilization  DOUBLE PRECISION,
    faithfulness         DOUBLE PRECISION,
    answer_relevancy     DOUBLE PRECISION,
    answer_correctness   DOUBLE PRECISION,
    semantic_similarity  DOUBLE PRECISION,

    question_score       DOUBLE PRECISION,  -- mean of this question's available metrics
    passed                BOOLEAN,
    latency_sec          DOUBLE PRECISION,
    token_usage           INTEGER,
    error_message          TEXT,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_score ON evaluation_results(run_id, question_score);
