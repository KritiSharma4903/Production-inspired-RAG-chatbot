from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_pc_client: Pinecone | None = None


def get_pinecone_client() -> Pinecone:
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc_client


def ensure_index_exists() -> None:
    """Idempotent: safe to call on every app startup. Creating an index is
    a one-time, slow (~1 min) operation, so we only do it if missing."""
    pc = get_pinecone_client()
    existing = [idx["name"] for idx in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )


def _index():
    return get_pinecone_client().Index(settings.PINECONE_INDEX_NAME)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def upsert_vectors(items: list[dict]) -> None:
    """
    items: [{"id": chunk_id, "values": [...], "metadata": {...}}, ...]

    Upsert is INHERENTLY IDEMPOTENT in Pinecone: upserting the same id
    twice overwrites in place, it never creates a duplicate. This is what
    makes safe retries possible -- if a worker crashes after upserting but
    before marking the job "succeeded" in Postgres, re-running the job just
    re-upserts the same vectors with no side effect.

    Batched in groups of 100 (Pinecone's recommended batch ceiling) to
    balance request overhead against payload size.
    """
    if not items:
        return
    index = _index()
    batch_size = 100
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        index.upsert(vectors=batch, namespace=settings.PINECONE_NAMESPACE)
    logger.info(f"Upserted {len(items)} vectors")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def delete_vectors(chunk_ids: list[str]) -> None:
    """Deletes by explicit id list -- used for the `deleted` bucket from the
    diff engine (chunks that existed in the old version but not the new
    one)."""
    if not chunk_ids:
        return
    index = _index()
    index.delete(ids=chunk_ids, namespace=settings.PINECONE_NAMESPACE)
    logger.info(f"Deleted {len(chunk_ids)} vectors")


def delete_all_for_document(document_id: str) -> None:
    """Bulk cleanup path -- e.g. when a user deletes a whole document, not
    just updates it. Filter-based delete avoids needing to enumerate every
    chunk_id client-side."""
    index = _index()
    index.delete(filter={"document_id": {"$eq": document_id}}, namespace=settings.PINECONE_NAMESPACE)


def query_similar(vector: list[float], top_k: int, document_id: str | None = None) -> list[dict]:
    """
    Retrieval-time ANN search. Optionally filtered to a single document_id
    (metadata filter) -- this is the same metadata field used for
    diffing/deletion, reused here for scoping a chat session to one
    document if the product requires it.
    """
    index = _index()
    query_filter = {"document_id": {"$eq": document_id}} if document_id else None
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=settings.PINECONE_NAMESPACE,
        filter=query_filter,
    )
    return [
        {
            "chunk_id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in result.get("matches", [])
    ]
