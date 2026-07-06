-- Optional users table (for multi-user support)
CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per logical document
CREATE TABLE IF NOT EXISTS documents (
    document_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Document name
    filename           TEXT NOT NULL,
    owner_user_id      UUID REFERENCES users(user_id),

    -- SHA-256 hash of the uploaded file
    latest_file_hash   TEXT NOT NULL,

    -- Current document version
    latest_version     INTEGER NOT NULL DEFAULT 1,

    -- Parser/chunker version
    parser_version     TEXT NOT NULL DEFAULT 'v1',

    total_pages        INTEGER,
    total_chunks       INTEGER NOT NULL DEFAULT 0,

    -- Current ingestion status
    ingestion_status   TEXT NOT NULL DEFAULT 'uploaded',
    last_error         TEXT,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(ingestion_status);

-- History of all document versions
CREATE TABLE IF NOT EXISTS document_versions (
    version_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version_number   INTEGER NOT NULL,
    file_hash        TEXT NOT NULL,
    storage_path     TEXT NOT NULL,
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    changed_chunks   INTEGER NOT NULL DEFAULT 0,
    deleted_chunks   INTEGER NOT NULL DEFAULT 0,
    inserted_chunks  INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, version_number)
);

-- Latest chunks for each document
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    -- SHA-256 hash of chunk content
    content_hash     TEXT NOT NULL,

    page_number      INTEGER,
    order_index      INTEGER NOT NULL,
    char_start       INTEGER,
    char_end         INTEGER,
    token_count      INTEGER,

    -- Chunk version information
    version_added    INTEGER NOT NULL,
    version_updated  INTEGER NOT NULL,

    -- Pinecone vector id
    vector_id        TEXT NOT NULL,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_version_updated
ON chunks(document_id, version_updated);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID REFERENCES users(user_id),
    title          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS chat_history (
    message_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id         UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role               TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content            TEXT NOT NULL,

    -- Retrieved chunk ids used for the answer
    retrieved_chunk_ids TEXT[],

    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_history_session
ON chat_history(session_id, created_at);

-- Background ingestion jobs
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    file_hash       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    attempts        INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, file_hash)
);
