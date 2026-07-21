"""
ragas_metrics.py
================
Adapter around the Ragas library. Wraps the project's EXISTING Groq LLM
(app/retrieval/llm_service.py) and embedding model (app/ingestion/embedder.py)
into the LangChain interfaces Ragas expects, instead of Ragas defaulting to
its own OpenAI client -- so eval runs use the same models the chatbot
actually runs on, and no second API key/provider is required.

Computes: context_precision, context_recall, faithfulness, answer_relevancy
(exactly the four metrics the brief asks for from Ragas).
"""

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from ragas import evaluate
from ragas.metrics import answer_relevancy as ragas_answer_relevancy
from ragas.metrics import context_precision as ragas_context_precision
from ragas.metrics import context_recall as ragas_context_recall
from ragas.metrics import faithfulness as ragas_faithfulness

from app.config import settings
from app.ingestion.embedder import embed_batch
from app.logging_config import get_logger
from app.retrieval.llm_service import generate_answer

logger = get_logger(__name__)


class _GroqLangChainLLM(LLM):
    """Minimal LangChain LLM shim over the project's existing Groq client
    so Ragas' judge calls route through the same model/config as the rest
    of the app (see app/config.py EVAL_JUDGE_MODEL)."""

    @property
    def _llm_type(self) -> str:
        return "groq-existing-client"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        return generate_answer(prompt)


class _ProjectEmbeddings(Embeddings):
    """Minimal LangChain Embeddings shim over the project's existing
    sentence-transformers model -- Ragas' context_precision/recall use
    embeddings internally for some sub-computations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_batch([text])[0]


def evaluate_with_ragas(
    question: str, answer: str, contexts: list[str], ground_truth: str | None = None
) -> dict:
    """
    Runs the four Ragas metrics for ONE question. Ragas is dataset-oriented
    (built for batch evaluation), so a single question is wrapped as a
    one-row HuggingFace Dataset -- evaluator.py's batch path
    (evaluate_dataset_with_ragas below) is the more efficient route for
    full experiment runs.

    context_recall and answer_correctness require ground_truth; Ragas
    returns NaN for rows missing it, which we normalize to None.
    """
    metrics = [ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy]
    row = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        row["ground_truth"] = [ground_truth]
        metrics.append(ragas_context_recall)

    dataset = Dataset.from_dict(row)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=_GroqLangChainLLM(),
        embeddings=_ProjectEmbeddings(),
    )
    scores = result.to_pandas().iloc[0].to_dict()

    return {
        "context_precision": _clean(scores.get("context_precision")),
        "context_recall": _clean(scores.get("context_recall")),
        "faithfulness": _clean(scores.get("faithfulness")),
        "answer_relevancy": _clean(scores.get("answer_relevancy")),
    }


def evaluate_dataset_with_ragas(rows: list[dict]) -> list[dict]:
    """
    Batch path -- evaluates an entire dataset in one Ragas call, which is
    both faster (Ragas internally batches its own LLM/embedding calls) and
    the intended usage pattern for `POST /evaluation/run` over a full
    dataset file (see dataset_loader.py). Each row must have
    question/answer/contexts, optionally ground_truth.
    """
    has_gt = all(r.get("ground_truth") for r in rows)
    metrics = [ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy]

    data = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    }
    if has_gt:
        data["ground_truth"] = [r["ground_truth"] for r in rows]
        metrics.append(ragas_context_recall)

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=metrics, llm=_GroqLangChainLLM(), embeddings=_ProjectEmbeddings())
    df = result.to_pandas()

    return [
        {
            "context_precision": _clean(row.get("context_precision")),
            "context_recall": _clean(row.get("context_recall")),
            "faithfulness": _clean(row.get("faithfulness")),
            "answer_relevancy": _clean(row.get("answer_relevancy")),
        }
        for _, row in df.iterrows()
    ]


def _clean(value) -> float | None:
    """Ragas returns NaN for metrics it couldn't compute (e.g. no ground
    truth for context_recall) -- normalize to None so it serializes to
    JSON `null` instead of an invalid `NaN` token."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f  # f != f is the NaN check
    except (TypeError, ValueError):
        return None
