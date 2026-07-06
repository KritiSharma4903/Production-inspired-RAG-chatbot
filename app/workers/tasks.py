import time
from concurrent.futures import ThreadPoolExecutor
from app.db.session import db_session_scope
from app.ingestion.pipeline import ingest_or_update_document
from app.logging_config import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)  # stand-in for N worker processes
MAX_ATTEMPTS = 3


def _run_with_retry(filename: str, file_bytes: bytes, owner_user_id: str | None):
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            with db_session_scope() as db:
                result = ingest_or_update_document(db, filename, file_bytes, owner_user_id)
            logger.info(f"Ingestion job succeeded for '{filename}' on attempt {attempt}: {result}")
            return result
        except Exception:
            logger.exception(f"Ingestion job failed for '{filename}' (attempt {attempt}/{MAX_ATTEMPTS})")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s, 8s
            else:
                logger.error(f"Ingestion job permanently failed for '{filename}' after {MAX_ATTEMPTS} attempts")
                raise


def enqueue_ingestion_job(filename: str, file_bytes: bytes, owner_user_id: str | None = None):
    """
    In-process stand-in for a real queue. Replace with, e.g.:

        # Celery
        from app.workers.celery_app import celery_app
        celery_app.send_task("ingest_document", args=[filename, file_bytes, owner_user_id])

        # or AWS SQS
        sqs.send_message(QueueUrl=..., MessageBody=json.dumps({...}))

    The API route would then only enqueue and return a job_id -- the actual
    `_run_with_retry` call would execute in a separate worker process pool,
    horizontally scalable independent of the API tier (see docs/
    ARCHITECTURE.md Part 11).
    """
    future = _executor.submit(_run_with_retry, filename, file_bytes, owner_user_id)
    return future
