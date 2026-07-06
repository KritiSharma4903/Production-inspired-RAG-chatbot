from app.config import settings
from app.ingestion.embedder import embed_query
from app.vectorstore.pinecone_client import query_similar


def retrieve_relevant_chunks(question: str, document_id: str | None = None, top_k: int | None = None) -> list[dict]:
    query_vector = embed_query(question)
    k = top_k or settings.TOP_K_RETRIEVAL
    matches = query_similar(query_vector, top_k=k, document_id=document_id)
    return matches
