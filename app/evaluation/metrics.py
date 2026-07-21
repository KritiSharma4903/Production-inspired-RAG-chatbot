"""
metrics.py
==========
Native (framework-free) implementations of every metric the brief asks for.
This is the DEFAULT evaluator -- it works with zero external eval
dependencies, using only the project's existing embedding model and Groq
client (via utils.py). Ragas/LangSmith/DeepEval are adapters layered on top
(see ragas_metrics.py, langsmith_evaluator.py) that can be swapped in via
config flags; this module is what runs if none of them are enabled or if
one of them errors out mid-run.

METRIC DEFINITIONS (also surfaced in docs/EVALUATION.md):

Retrieval metrics
------------------
- context_precision: of the retrieved chunks, what fraction are actually
  relevant to the question? (signal-to-noise of the retriever)
- context_recall: of the information needed to answer (per the ground
  truth), what fraction was actually retrieved? (did we retrieve ENOUGH)
- context_relevancy: embedding-similarity between the question and each
  retrieved chunk, averaged -- a cheap, LLM-free proxy for precision.
- context_utilization: of the retrieved chunks, what fraction did the
  generated answer actually draw on? (are we retrieving chunks the LLM
  then ignores -- wasted retrieval)

Generation metrics
-------------------
- faithfulness: is every claim in the generated answer actually supported
  by the retrieved context? (hallucination detector)
- answer_relevancy: does the answer actually address the question asked?
  (an answer can be faithful to context yet off-topic)
- answer_correctness: does the answer match the ground truth answer?
  (requires a ground truth; faithfulness/relevancy do not)
- semantic_similarity: embedding cosine similarity between generated
  answer and ground truth -- a cheap, LLM-free proxy for correctness.
"""

from app.evaluation.utils import cosine_similarity, embed_texts, llm_judge_json
from app.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------- Retrieval metrics ---------------------------

def context_relevancy(question: str, contexts: list[str]) -> float:
    """Embedding-only, no LLM call: mean cosine similarity between the
    question and each retrieved chunk. Cheap enough to run on every query
    in production (not just eval runs) as a live retrieval-quality signal."""
    if not contexts:
        return 0.0
    q_vec = embed_texts([question])[0]
    c_vecs = embed_texts(contexts)
    sims = [cosine_similarity(q_vec, c) for c in c_vecs]
    return sum(sims) / len(sims)


def context_precision(question: str, contexts: list[str], ground_truth: str | None) -> float:
    """LLM-as-judge: for each chunk, ask 'was this necessary to answer the
    question (given the ground truth)?' Precision = relevant / retrieved.
    Falls back to context_relevancy (embedding-based) if no ground truth
    is supplied, since precision-against-nothing isn't well-defined."""
    if not contexts:
        return 0.0
    if not ground_truth:
        return context_relevancy(question, contexts)

    relevant = 0
    for ctx in contexts:
        prompt = (
            "You are grading retrieval quality. Given a question, a ground-truth "
            "answer, and ONE retrieved passage, respond with strict JSON only: "
            '{"relevant": true|false}. A passage is relevant if it contains '
            "information that helps produce the ground-truth answer.\n\n"
            f"Question: {question}\nGround truth: {ground_truth}\nPassage: {ctx}"
        )
        verdict = llm_judge_json(prompt)
        if verdict.get("relevant") is True:
            relevant += 1
    return relevant / len(contexts)


def context_recall(contexts: list[str], ground_truth: str | None) -> float:
    """LLM-as-judge: what fraction of the CLAIMS in the ground truth answer
    are supported by at least one retrieved chunk? Requires ground truth --
    returns None (surfaced as null in the API) if none is supplied, since
    recall is undefined without a target to recall against."""
    if not ground_truth or not contexts:
        return 0.0 if ground_truth else None

    context_blob = "\n---\n".join(contexts)
    prompt = (
        "You are grading retrieval recall. Break the ground-truth answer into "
        "atomic factual claims, then determine what fraction of those claims are "
        "supported by the provided context. Respond with strict JSON only: "
        '{"recall": <float 0.0-1.0>}.\n\n'
        f"Ground truth answer: {ground_truth}\n\nContext:\n{context_blob}"
    )
    verdict = llm_judge_json(prompt)
    try:
        return max(0.0, min(1.0, float(verdict.get("recall", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def context_utilization(answer: str, contexts: list[str]) -> float:
    """LLM-as-judge: of the retrieved chunks, how many does the generated
    answer actually draw on? Distinguishes 'retriever pulled irrelevant
    chunks' (low context_precision) from 'retriever pulled fine chunks but
    the LLM ignored most of them' (low context_utilization, high
    context_precision) -- two different bugs with different fixes."""
    if not contexts:
        return 0.0
    used = 0
    for ctx in contexts:
        prompt = (
            "Does the ANSWER below draw on information from the PASSAGE? Respond "
            'with strict JSON only: {"used": true|false}.\n\n'
            f"Passage: {ctx}\n\nAnswer: {answer}"
        )
        verdict = llm_judge_json(prompt)
        if verdict.get("used") is True:
            used += 1
    return used / len(contexts)


# --------------------------- Generation metrics ---------------------------

def faithfulness(answer: str, contexts: list[str]) -> float:
    """LLM-as-judge: decompose the answer into claims, check each against
    the retrieved context. This is the primary hallucination detector --
    an answer can score well on answer_relevancy while being unfaithful
    (confidently wrong, but on-topic)."""
    if not contexts:
        return 0.0
    context_blob = "\n---\n".join(contexts)
    prompt = (
        "Break the ANSWER into atomic factual claims. For each claim, check "
        "whether it is directly supported by the CONTEXT. Respond with strict "
        'JSON only: {"faithfulness": <float 0.0-1.0>} representing the fraction '
        "of claims that are supported.\n\n"
        f"Context:\n{context_blob}\n\nAnswer: {answer}"
    )
    verdict = llm_judge_json(prompt)
    try:
        return max(0.0, min(1.0, float(verdict.get("faithfulness", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def answer_relevancy(question: str, answer: str) -> float:
    """Embedding-based (Ragas-style): generate N pseudo-questions the given
    answer would be answering, embed them, compare to the actual question.
    Simplified here to a direct question<->answer embedding similarity,
    which is a reasonable cheap proxy and avoids an extra LLM round-trip
    per evaluation (documented trade-off vs. the full Ragas method)."""
    if not answer.strip():
        return 0.0
    q_vec, a_vec = embed_texts([question, answer])
    return cosine_similarity(q_vec, a_vec)


def answer_correctness(answer: str, ground_truth: str | None) -> float | None:
    """LLM-as-judge: does the answer factually match the ground truth?
    Different from faithfulness (which checks answer-vs-context) and from
    answer_relevancy (which checks answer-vs-question) -- correctness is
    the only metric that checks answer-vs-KNOWN-RIGHT-ANSWER, so it's the
    only one that can catch 'faithful to context, on-topic, but still
    wrong because the context itself didn't fully cover the answer.'"""
    if not ground_truth:
        return None
    prompt = (
        "Compare the GENERATED ANSWER to the GROUND TRUTH ANSWER for factual "
        "correctness (ignore phrasing/style differences). Respond with strict "
        'JSON only: {"correctness": <float 0.0-1.0>}.\n\n'
        f"Ground truth: {ground_truth}\n\nGenerated answer: {answer}"
    )
    verdict = llm_judge_json(prompt)
    try:
        return max(0.0, min(1.0, float(verdict.get("correctness", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def semantic_similarity(answer: str, ground_truth: str | None) -> float | None:
    """Embedding-only, no LLM call: cosine similarity between generated
    answer and ground truth. Cheap proxy for answer_correctness -- useful
    for fast/frequent eval runs where LLM-judge cost/latency matters."""
    if not ground_truth:
        return None
    a_vec, g_vec = embed_texts([answer, ground_truth])
    return cosine_similarity(a_vec, g_vec)


def evaluate_native(
    question: str, answer: str, contexts: list[str], ground_truth: str | None = None
) -> dict:
    """Runs every native metric for one question and returns a flat dict
    matching EvaluationResponse's fields. This is the function evaluator.py
    calls when RAGAS_ENABLED/LANGSMITH_ENABLED/DEEPEVAL_ENABLED are all
    False, or as a per-question fallback if an adapter call raises."""
    return {
        "context_precision": context_precision(question, contexts, ground_truth),
        "context_recall": context_recall(contexts, ground_truth),
        "context_relevancy": context_relevancy(question, contexts),
        "context_utilization": context_utilization(answer, contexts),
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevancy": answer_relevancy(question, answer),
        "answer_correctness": answer_correctness(answer, ground_truth),
        "semantic_similarity": semantic_similarity(answer, ground_truth),
    }
