from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.ingestion.pipeline import ingest_or_update_document
from app.logging_config import get_logger

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_extensions = [".pdf", ".docx"]

    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed"
        )
    
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = ingest_or_update_document(db=db, filename=file.filename, file_bytes=content)
    except Exception as e:
        logger.exception(f"Ingestion failed for '{file.filename}'")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return result


@router.get("/{document_id}/status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    from app.db.models import Document
    doc = db.query(Document).filter(Document.document_id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": str(doc.document_id),
        "filename": doc.filename,
        "status": doc.ingestion_status,
        "version": doc.latest_version,
        "total_chunks": doc.total_chunks,
        "last_error": doc.last_error,
    }


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    from app.db.models import Document
    from app.vectorstore.pinecone_client import delete_all_for_document

    doc = db.query(Document).filter(Document.document_id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_all_for_document(document_id)  # Pinecone cleanup first
    db.delete(doc)                        # cascades to chunks/versions via FK
    db.commit()
    return {"status": "deleted", "document_id": document_id}


