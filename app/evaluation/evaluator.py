import time
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import EvaluationRun, EvaluationResult
from app.evaluation import metrics as native_metrics
from app.evaluation.dataset_loader import DatasetExample
from app.evaluation.utils import timer
from app.logging_config import get_logger
from app.retrieval.llm_service import generate_answer
from app.retrieval.prompt_builder import build_prompt
from app.retrieval.retriever import retrieve_relevant_chunks

logger = get_logger(__name__)

PASS_THRESHOLD = 0.6  # a question "passes" if its mean available-metric score clears this


def _select_adapter():
    """Returns (name, fn) where fn(question, answer, contexts, ground_truth) -> dict."""
    if settings.RAGAS_ENABLED:
        from app.evaluation.ragas_metrics import evaluate_with_ragas
        return "ragas", evaluate_with_ragas
    if settings.DEEPEVAL_ENABLED:
        from app.evaluation.deepeval_adapter import evaluate_with_deepeval
        return "deepeval", evaluate_with_deepeval
    if settings.LANGSMITH_ENABLED:
        # LangSmith is experiment/tracing-oriented rather than a per-call
        # scorer; per-question scoring still uses native metrics, with
        # results additionally traced to LangSmith (see routes_evaluation.py).
        return "native+langsmith", native_metrics.evaluate_native
    return "native", native_metrics.evaluate_native


def run_pipeline_for_question(question: str, document_id: str | None = None) -> dict:
    """Runs ONLY the retrieve+generate steps (no scoring) -- this is the
    exact code path `chat/ask` uses, reused here so eval measures
    production behavior. Returns {"answer": ..., "contexts": [...]}."""
    matches = retrieve_relevant_chunks(question, document_id=document_id)
    contexts = [m["metadata"].get("text", "") for m in matches]
    prompt = build_prompt(question, matches)
    answer = generate_answer(prompt)
    return {"answer": answer, "contexts": contexts}


def evaluate_single(
    question: str, ground_truth: str | None = None, document_id: str | None = None,
    answer: str | None = None, contexts: list[str] | None = None,
) -> dict:
    """
    Evaluates ONE question end-to-end. If `answer`/`contexts` are already
    provided (e.g. from a prior chat turn being retroactively scored), the
    retrieve+generate step is skipped -- otherwise it's run fresh.
    """
    adapter_name, adapter_fn = _select_adapter()

    with timer() as t:
        if answer is None or contexts is None:
            generated = run_pipeline_for_question(question, document_id)
            answer = generated["answer"]
            contexts = generated["contexts"]

        try:
            scores = adapter_fn(question, answer, contexts, ground_truth)
        except Exception:
            logger.exception(f"{adapter_name} adapter failed for question, falling back to native")
            adapter_name = "native (fallback)"
            scores = native_metrics.evaluate_native(question, answer, contexts, ground_truth)

    available = [v for v in scores.values() if isinstance(v, (int, float))]
    question_score = sum(available) / len(available) if available else 0.0

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        **scores,
        "question_score": question_score,
        "passed": question_score >= PASS_THRESHOLD,
        "latency_sec": t["elapsed"],
        "evaluator_used": adapter_name,
    }


def run_dataset_evaluation(
    db: Session, examples: list[DatasetExample], experiment_name: str, dataset_name: str,
    document_id: str | None = None,
) -> str:
    """
    Batch-evaluates a full dataset, persists an EvaluationRun + one
    EvaluationResult per question, and returns the run_id. This is what
    `POST /evaluation/run` calls.
    """
    run = EvaluationRun(
        run_id=uuid.uuid4(), experiment_name=experiment_name, dataset_name=dataset_name,
        evaluator_used=_select_adapter()[0], total_questions=len(examples),
        started_at=datetime.now(timezone.utc), status="running",
    )
    db.add(run)
    db.flush()

    overall_start = time.perf_counter()
    per_question_results = []

    try:
        for ex in examples:
            result = evaluate_single(ex.question, ex.ground_truth, document_id)
            per_question_results.append(result)

            db.add(EvaluationResult(
                run_id=run.run_id, question=result["question"], generated_answer=result["answer"],
                ground_truth=result["ground_truth"], retrieved_contexts=result["contexts"],
                context_precision=result.get("context_precision"), context_recall=result.get("context_recall"),
                context_relevancy=result.get("context_relevancy"), context_utilization=result.get("context_utilization"),
                faithfulness=result.get("faithfulness"), answer_relevancy=result.get("answer_relevancy"),
                answer_correctness=result.get("answer_correctness"), semantic_similarity=result.get("semantic_similarity"),
                question_score=result["question_score"], passed=result["passed"], latency_sec=result["latency_sec"],
            ))

        _finalize_run(run, per_question_results, time.perf_counter() - overall_start)
        run.status = "completed"

    except Exception as e:
        logger.exception(f"Evaluation run {run.run_id} failed")
        run.status = "failed"
        run.error_message = str(e)

    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    return str(run.run_id)


def _finalize_run(run: EvaluationRun, results: list[dict], elapsed: float) -> None:
    def avg(key: str) -> float | None:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    run.avg_context_precision = avg("context_precision")
    run.avg_context_recall = avg("context_recall")
    run.avg_context_relevancy = avg("context_relevancy")
    run.avg_context_utilization = avg("context_utilization")
    run.avg_faithfulness = avg("faithfulness")
    run.avg_answer_relevancy = avg("answer_relevancy")
    run.avg_answer_correctness = avg("answer_correctness")
    run.avg_semantic_similarity = avg("semantic_similarity")

    metric_avgs = [
        run.avg_context_precision, run.avg_context_recall, run.avg_context_relevancy,
        run.avg_context_utilization, run.avg_faithfulness, run.avg_answer_relevancy,
        run.avg_answer_correctness, run.avg_semantic_similarity,
    ]
    available = [m for m in metric_avgs if m is not None]
    run.overall_score = sum(available) / len(available) if available else None

    run.pass_count = sum(1 for r in results if r["passed"])
    run.fail_count = len(results) - run.pass_count
    run.execution_time_sec = elapsed


