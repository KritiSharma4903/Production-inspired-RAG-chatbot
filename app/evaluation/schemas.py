"""
schemas.py
==========
WHY THIS FILE EXISTS:
Every request/response shape crossing the `/evaluation/*` HTTP boundary is
defined ONCE here -- `routes_evaluation.py` imports these instead of
building raw dicts, so FastAPI gets automatic request validation and OpenAPI
docs for free, and `service.py` has one canonical shape to fill in whether
the score came from Ragas, DeepEval, LangSmith, or the native fallback.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    question: str
    # Both optional: if omitted, evaluate_single() runs the live
    # retrieve+generate pipeline itself (see evaluator.py). Supply both to
    # instead score an already-generated answer without re-running the LLM.
    answer: str | None = None
    contexts: list[str] | None = None
    ground_truth: str | None = None
    document_id: str | None = None
    # Optional per-request override of which backend to use -- if omitted,
    # the Evaluator uses whichever *_ENABLED flag is set in config (see
    # evaluator.py). Lets a caller A/B two adapters without touching env vars.
    evaluator: str | None = None


class EvaluationResponse(BaseModel):
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    correctness: float | None = None
    latency: float
    evaluator_used: str


class EvaluationHistoryItem(BaseModel):
    id: str
    timestamp: datetime
    document_id: str | None
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None


class DatasetExampleIn(BaseModel):
    question: str
    ground_truth: str | None = None
    expected_context: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DatasetCreateRequest(BaseModel):
    dataset_name: str
    examples: list[DatasetExampleIn]


class RunEvaluationRequest(BaseModel):
    experiment_name: str
    dataset_name: str          # filename under evaluation_data/, e.g. "sample_dataset.json"
    document_id: str | None = None


class RunEvaluationResponse(BaseModel):
    run_id: str
    status: str
    total_questions: int


class RunSummary(BaseModel):
    run_id: str
    experiment_name: str
    dataset_name: str
    evaluator_used: str
    overall_score: float | None
    pass_count: int
    fail_count: int
    status: str
    started_at: datetime
    finished_at: datetime | None
    correctness: float | None
    latency: float
    evaluator_used: str

    model_config = {"from_attributes": True}


class MetricsAverageResponse(BaseModel):
    total_evaluations: int
    avg_context_precision: float | None
    avg_context_recall: float | None
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_correctness: float | None
    avg_latency: float | None
