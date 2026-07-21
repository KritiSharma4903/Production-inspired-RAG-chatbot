"""
langsmith_evaluator.py
=======================
LangSmith integration: tracing setup, dataset/example management, running
evaluations, custom evaluators, and comparing experiments.

Tracing is configured by exporting LANGCHAIN_* env vars at import time
(LangChain's SDK auto-instruments from these -- there's no explicit
"start tracing" call needed once they're set). Every `generate_answer()`
call elsewhere in the app is unaffected by whether this module is imported;
tracing is opt-in via LANGSMITH_ENABLED.
"""

import os
import time

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from langsmith.schemas import Example, Run

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_client: Client | None = None


def configure_tracing() -> None:
    """Call once at startup if LANGSMITH_ENABLED. Sets the env vars
    LangChain's SDK reads for auto-tracing -- after this, LLM calls made
    through LangChain-wrapped clients elsewhere are traced automatically."""
    if not settings.LANGSMITH_ENABLED:
        return
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    logger.info(f"LangSmith tracing configured for project '{settings.LANGCHAIN_PROJECT}'")


def get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(api_key=settings.LANGCHAIN_API_KEY)
    return _client


def create_dataset(dataset_name: str, description: str = "") -> str:
    """Idempotent: reuses an existing dataset of the same name instead of
    erroring, so re-running ingestion of the same eval dataset file is safe."""
    client = get_client()
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        return str(existing[0].id)
    dataset = client.create_dataset(dataset_name=dataset_name, description=description)
    return str(dataset.id)


def add_examples(dataset_name: str, rows: list[dict]) -> int:
    """
    rows: [{"question": ..., "ground_truth": ..., "expected_context": [...], "metadata": {...}}, ...]
    Each row becomes one LangSmith Example (inputs=question, outputs=ground_truth).
    Returns the number of examples created.
    """
    dataset_id = create_dataset(dataset_name)
    client = get_client()
    count = 0
    for row in rows:
        client.create_example(
            dataset_id=dataset_id,
            inputs={"question": row["question"]},
            outputs={"ground_truth": row.get("ground_truth", "")},
            metadata={
                "expected_context": row.get("expected_context", []),
                **row.get("metadata", {}),
            },
        )
        count += 1
    return count


# --------------------------- Custom evaluators ---------------------------

def _faithfulness_evaluator(run: Run, example: Example) -> dict:
    """LangSmith custom evaluator function -- signature is (run, example) ->
    {"key": ..., "score": ...} per LangSmith's evaluator contract. Reuses
    the project's own faithfulness metric rather than a LangSmith built-in,
    so scores are comparable to the Ragas/native adapters' output."""
    from app.evaluation.metrics import faithfulness as native_faithfulness

    answer = run.outputs.get("answer", "") if run.outputs else ""
    contexts = run.outputs.get("contexts", []) if run.outputs else []
    score = native_faithfulness(answer, contexts)
    return {"key": "faithfulness", "score": score}


def _correctness_evaluator(run: Run, example: Example) -> dict:
    from app.evaluation.metrics import answer_correctness as native_correctness

    answer = run.outputs.get("answer", "") if run.outputs else ""
    ground_truth = example.outputs.get("ground_truth") if example.outputs else None
    score = native_correctness(answer, ground_truth) or 0.0
    return {"key": "correctness", "score": score}


def run_experiment(dataset_name: str, target_fn, experiment_prefix: str = "rag-eval"):
    """
    target_fn: callable(inputs: dict) -> dict, e.g. a wrapper around
    evaluator.py's per-question pipeline that takes {"question": ...} and
    returns {"answer": ..., "contexts": [...]}.

    Runs `target_fn` over every example in the named dataset, scores each
    with the custom evaluators above, and records the run as a LangSmith
    "experiment" -- experiments with the same dataset can be compared via
    compare_experiments() below.
    """
    configure_tracing()
    start = time.perf_counter()
    results = ls_evaluate(
        target_fn,
        data=dataset_name,
        evaluators=[_faithfulness_evaluator, _correctness_evaluator],
        experiment_prefix=experiment_prefix,
    )
    elapsed = time.perf_counter() - start
    logger.info(f"LangSmith experiment '{experiment_prefix}' completed in {elapsed:.2f}s")
    return results


def compare_experiments(dataset_name: str, experiment_names: list[str]) -> dict:
    """
    Pulls aggregate feedback scores for each named experiment run against
    the same dataset and returns a side-by-side comparison dict --
    {"faithfulness": {"exp_a": 0.81, "exp_b": 0.88}, ...}. Use this to
    answer "did prompt change v2 actually improve faithfulness over v1?"
    """
    client = get_client()
    comparison: dict[str, dict[str, float]] = {}
    for exp_name in experiment_names:
        runs = list(client.list_runs(project_name=exp_name, execution_order=1))
        feedback = client.list_feedback(run_ids=[r.id for r in runs])
        per_key: dict[str, list[float]] = {}
        for fb in feedback:
            per_key.setdefault(fb.key, []).append(fb.score or 0.0)
        for key, scores in per_key.items():
            comparison.setdefault(key, {})[exp_name] = sum(scores) / len(scores) if scores else 0.0
    return comparison

