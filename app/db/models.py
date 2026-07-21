import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Text, TIMESTAMP, ForeignKey, ARRAY, UniqueConstraint, func, Float, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    """
    Optional table (per the brief) -- only needed if you want per-tenant
    document isolation and multi-user chat history. Defined here (not just
    in sql/schema.sql) because SQLAlchemy's `Base.metadata.create_all()`
    only knows about tables that have a corresponding ORM class registered
    against this same `Base` -- it does NOT read sql/schema.sql. Any table
    referenced via `ForeignKey("some_table.column")` must have a matching
    class here, or table creation fails with NoReferencedTableError (which
    is exactly the error you hit).
    """
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    latest_file_hash = Column(Text, nullable=False)
    latest_version = Column(Integer, nullable=False, default=1)
    parser_version = Column(Text, nullable=False, default="v1")
    total_pages = Column(Integer)
    total_chunks = Column(Integer, nullable=False, default=0)
    ingestion_status = Column(Text, nullable=False, default="uploaded")
    last_error = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"))
    version_number = Column(Integer, nullable=False)
    file_hash = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    chunk_count = Column(Integer, default=0)
    changed_chunks = Column(Integer, default=0)
    deleted_chunks = Column(Integer, default=0)
    inserted_chunks = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="versions")

    __table_args__ = (UniqueConstraint("document_id", "version_number"),)


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(Text, primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"))
    content_hash = Column(Text, nullable=False)
    page_number = Column(Integer)
    order_index = Column(Integer, nullable=False)
    char_start = Column(Integer)
    char_end = Column(Integer)
    token_count = Column(Integer)
    version_added = Column(Integer, nullable=False)
    version_updated = Column(Integer, nullable=False)
    vector_id = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    title = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class ChatHistory(Base):
    __tablename__ = "chat_history"

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"))
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(ARRAY(Text))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"))
    file_hash = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="queued")
    attempts = Column(Integer, nullable=False, default=0)
    error_message = Column(Text)
    started_at = Column(TIMESTAMP(timezone=True))
    finished_at = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("document_id", "file_hash"),)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_name = Column(Text, nullable=False)
    dataset_name = Column(Text, nullable=False)
    evaluator_used = Column(Text, nullable=False)
    total_questions = Column(Integer, nullable=False, default=0)

    avg_context_precision = Column(Float)
    avg_context_recall = Column(Float)
    avg_context_relevancy = Column(Float)
    avg_context_utilization = Column(Float)
    avg_faithfulness = Column(Float)
    avg_answer_relevancy = Column(Float)
    avg_answer_correctness = Column(Float)
    avg_semantic_similarity = Column(Float)
    overall_score = Column(Float)

    pass_count = Column(Integer, nullable=False, default=0)
    fail_count = Column(Integer, nullable=False, default=0)
    execution_time_sec = Column(Float)

    status = Column(Text, nullable=False, default="running")
    error_message = Column(Text)

    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    finished_at = Column(TIMESTAMP(timezone=True))

    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_runs.run_id", ondelete="CASCADE"))

    question = Column(Text, nullable=False)
    generated_answer = Column(Text)
    ground_truth = Column(Text)
    retrieved_contexts = Column(ARRAY(Text))

    context_precision = Column(Float)
    context_recall = Column(Float)
    context_relevancy = Column(Float)
    context_utilization = Column(Float)
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    answer_correctness = Column(Float)
    semantic_similarity = Column(Float)

    question_score = Column(Float)
    passed = Column(Boolean)
    latency_sec = Column(Float)
    token_usage = Column(Integer)
    error_message = Column(Text)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    run = relationship("EvaluationRun", back_populates="results")
