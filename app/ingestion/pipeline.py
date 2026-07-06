import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.models import Document, DocumentVersion, Chunk as ChunkRow, IngestionJob
from app.ingestion.chunker import chunk_document, Chunk
from app.ingestion.diff_engine import diff_chunk_sets, build_hash_map, DiffResult
from app.ingestion.embedder import embed_batch
from app.ingestion.parser import parse_pdf, parse_docx
from app.logging_config import get_logger
from app.storage.file_storage import get_file_storage, compute_file_hash
from app.vectorstore.pinecone_client import upsert_vectors, delete_vectors

logger = get_logger(__name__)

def _load_old_hash_map(db: Session, document_id: str) -> dict[str, str]:
    """Reads the CURRENT hash table for a document from Postgres -- this is
    the `old` side of the diff. O(chunk_count) row reads, indexed by
    document_id (see idx_chunks_document_id)."""
    rows = db.query(ChunkRow.chunk_id, ChunkRow.content_hash).filter(
        ChunkRow.document_id == document_id
    ).all()
    return {chunk_id: content_hash for chunk_id, content_hash in rows}


MAX_METADATA_TEXT_CHARS = 2000  # Pinecone metadata payload limits apply per-vector; keep this conservative


def _build_vector_payload(chunk: Chunk, document_id: str, filename: str, version: int, vector: list[float]) -> dict:
    return {
        "id": chunk.chunk_id,
        "values": vector,
        "metadata": {
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "page": chunk.page_number,
            "version": version,
            "content_hash": chunk.content_hash,
            "filename": filename,
            "text": chunk.text[:MAX_METADATA_TEXT_CHARS],
        },
    }


def ingest_or_update_document(db: Session, filename: str, file_bytes: bytes, owner_user_id: str | None = None) -> dict:
    """
    Single entry point for BOTH initial ingestion (Part 5) and updates
    (Part 6) -- a first-time upload is just "a diff against an empty old
    hash map," so we don't need two separate code paths. This symmetry is
    intentional: it eliminates an entire class of bugs where the update
    path silently diverges from the ingestion path.
    """
    new_file_hash = compute_file_hash(file_bytes)

    # --- Step 1: does this logical document already exist? ---
    existing_doc = db.query(Document).filter(Document.filename == filename).first()

    if existing_doc and existing_doc.latest_file_hash == new_file_hash:
        # --- Idempotency gate #1: identical file re-uploaded, no-op. ---
        logger.info(f"'{filename}' unchanged (file hash match) -- skipping reprocessing")
        return {"status": "unchanged", "document_id": str(existing_doc.document_id), "version": existing_doc.latest_version}

    if existing_doc:
        document_id = str(existing_doc.document_id)
        new_version = existing_doc.latest_version + 1
    else:
        document_id = str(uuid.uuid4())
        new_version = 1

    # --- Idempotency gate #2: has THIS exact (document_id, file_hash) job
    # already been recorded? Guards against duplicate concurrent submits
    # (e.g. a user double-clicking "upload"). ---
    job = db.query(IngestionJob).filter_by(document_id=document_id, file_hash=new_file_hash).first()
    if job and job.status in ("running", "succeeded"):
        logger.info("Duplicate ingestion job detected -- skipping")
        return {"status": job.status, "document_id": document_id}

    if not job:
        job = IngestionJob(
            document_id=uuid.UUID(document_id),
            file_hash=new_file_hash,
            status="running",
            started_at=datetime.now(timezone.utc)
        )
        # if this is a brand-new document, job.document_id must match the doc
        # we're about to create below -- handled by flush ordering.

    storage = get_file_storage()
    storage_path = storage.save(document_id, new_version, filename, file_bytes)

    # --- Step 2: parse (always full, cheap, no API cost) ---
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        blocks = parse_pdf(file_bytes)

    elif filename_lower.endswith(".docx"):
        blocks = parse_docx(file_bytes)

    else:
        raise ValueError(f"Unsupported file type: {filename}")

    # --- Step 3: chunk (always full; produces deterministic content-hashed ids) ---
    new_chunks = chunk_document(blocks, document_id)
    new_hash_map = build_hash_map(new_chunks)

    # --- Step 4: diff against Postgres hash table (empty dict for new docs) ---
    old_hash_map = _load_old_hash_map(db, document_id) if existing_doc else {}
    diff: DiffResult = diff_chunk_sets(old_hash_map, new_hash_map)

    logger.info(
        f"Diff for '{filename}': unchanged={len(diff.unchanged)} changed={len(diff.changed)} "
        f"deleted={len(diff.deleted)} inserted={len(diff.inserted)}"
    )

    # --- Step 5: embed ONLY changed+inserted chunks (the cost-saving step) ---
    chunks_by_id = {c.chunk_id: c for c in new_chunks}
    to_embed_ids = diff.to_embed
    texts = [chunks_by_id[cid].text for cid in to_embed_ids]
    vectors = embed_batch(texts)

    # --- Step 6: upsert changed+inserted vectors into Pinecone ---
    payloads = [
        _build_vector_payload(chunks_by_id[cid], document_id, filename, new_version, vec)
        for cid, vec in zip(to_embed_ids, vectors)
    ]
    upsert_vectors(payloads)

    # --- Step 7: delete removed vectors from Pinecone ---
    delete_vectors(diff.deleted)

    # --- Step 8: synchronize Postgres metadata ---
    # 8a. remove rows for deleted chunk_ids
    if diff.deleted:
        db.query(ChunkRow).filter(ChunkRow.chunk_id.in_(diff.deleted)).delete(synchronize_session=False)

    # 8b. upsert rows for changed+inserted chunk_ids (full metadata refresh)
    for cid in to_embed_ids:
        c = chunks_by_id[cid]
        existing_row = db.get(ChunkRow, cid)
        if existing_row:
            existing_row.content_hash = c.content_hash
            existing_row.page_number = c.page_number
            existing_row.order_index = c.order_index
            existing_row.char_start = c.char_start
            existing_row.char_end = c.char_end
            existing_row.token_count = c.token_count
            existing_row.version_updated = new_version
        else:
            db.add(ChunkRow(
                chunk_id=c.chunk_id, document_id=document_id, content_hash=c.content_hash,
                page_number=c.page_number, order_index=c.order_index, char_start=c.char_start,
                char_end=c.char_end, token_count=c.token_count, version_added=new_version,
                version_updated=new_version, vector_id=c.chunk_id,
            ))

    # 8c. metadata-only patch for unchanged-but-relocated chunks (position
    # may have shifted even though content_hash didn't -- cheap DB update,
    # NO re-embedding, NO Pinecone vector write beyond its existing metadata
    # if you choose to also patch Pinecone's stored `page` field here).
    for cid in diff.unchanged:
        c = chunks_by_id[cid]
        row = db.get(ChunkRow, cid)
        if row and (row.page_number != c.page_number or row.order_index != c.order_index):
            row.page_number = c.page_number
            row.order_index = c.order_index

    # 8d. upsert the documents + document_versions rows
    if existing_doc:
        existing_doc.latest_file_hash = new_file_hash
        existing_doc.latest_version = new_version
        existing_doc.total_pages = max((b.page_number for b in blocks), default=0)
        existing_doc.total_chunks = len(new_chunks)
        existing_doc.ingestion_status = "ready"
        doc_row = existing_doc
    else:
        doc_row = Document(
            document_id=document_id, filename=filename, owner_user_id=owner_user_id,
            latest_file_hash=new_file_hash, latest_version=new_version,
            total_pages=max((b.page_number for b in blocks), default=0),
            total_chunks=len(new_chunks), ingestion_status="ready",
        )
        db.add(doc_row)
        db.flush()

    db.add(DocumentVersion(
        document_id=document_id, version_number=new_version, file_hash=new_file_hash,
        storage_path=storage_path, chunk_count=len(new_chunks),
        changed_chunks=len(diff.changed), deleted_chunks=len(diff.deleted),
        inserted_chunks=len(diff.inserted),
    ))

    job.status = "succeeded"
    job.finished_at = datetime.now(timezone.utc)
    db.add(job)

    db.commit()

    return {
        "status": "ready",
        "document_id": document_id,
        "version": new_version,
        "chunks_total": len(new_chunks),
        "chunks_embedded": len(to_embed_ids),
        "chunks_deleted": len(diff.deleted),
        "chunks_unchanged": len(diff.unchanged),
    }
