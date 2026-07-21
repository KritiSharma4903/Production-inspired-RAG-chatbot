"""
utils.py
========
WHY THIS FILE EXISTS:
`native metrics`, `ragas_adapter`, `deepeval_adapter`, and `langsmith_adapter`
all need the same three primitives: (1) turn text into vectors using the
project's EXISTING embedding model, (2) compute cosine similarity, (3) call
the project's EXISTING Groq LLM service and parse a JSON verdict out of it.
Centralizing these here means every adapter reuses the same embedding model
and the same LLM client already used by the chatbot -- no adapter
introduces a second, uncontrolled model dependency, and none of them
duplicate this logic (per the brief's "do not duplicate existing code").
"""

import json
import re
import time
from contextlib import contextmanager

from app.ingestion.embedder import embed_batch
from app.logging_config import get_logger
from app.retrieval.llm_service import generate_answer

logger = get_logger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Reuses the SAME sentence-transformers model already loaded for
    ingestion/retrieval (app/ingestion/embedder.py) -- no second embedding
    model is loaded anywhere in the evaluation module."""
    if not texts:
        return []
    return embed_batch(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def llm_judge_json(prompt: str) -> dict:
    """
    Calls the project's existing Groq client (app/retrieval/llm_service.py)
    with a prompt that demands strict JSON, and parses it defensively.
    Every adapter's LLM-as-judge calls route through this one function so
    a malformed judge response degrades to a score of 0.0 with a logged
    reason instead of crashing the whole evaluation run.
    """
    raw = generate_answer(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning(f"LLM judge returned non-JSON output: {raw[:200]}")
        return {"score": 0.0, "reason": f"Unparseable judge response: {raw[:200]}"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning(f"LLM judge returned malformed JSON: {match.group(0)[:200]}")
        return {"score": 0.0, "reason": "Malformed JSON from judge"}


@contextmanager
def timer():
    """Usage: `with timer() as t: ...; t.elapsed` -- used by service.py to
    measure end-to-end evaluation latency for the `latency` column/field."""
    start = time.perf_counter()
    box = {"elapsed": 0.0}
    try:
        yield box
    finally:
        box["elapsed"] = time.perf_counter() - start
