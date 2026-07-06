from functools import lru_cache
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Loaded once per process (model load is slow) and reused across every
    # request/worker task -- NOT reloaded per-document.
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of chunk texts. Retried up to 3 times with exponential
    backoff (1s, 2s, 4s) to absorb transient failures (rate limits, network
    blips) WITHOUT failing the whole ingestion job over one flaky call.

    This is called ONLY with the diff'd subset of chunks (changed + newly
    inserted) -- never the full chunk set on an update. That gating happens
    one layer up, in pipeline.py, which is the crux of the cost savings
    described in docs/COST_ANALYSIS.md.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Single-text embedding path used at query time (Section 7 chat
    pipeline). Kept separate from embed_batch for clarity even though the
    underlying call is the same -- query embeddings are latency-critical
    and never batched with ingestion traffic."""
    return embed_batch([text])[0]
