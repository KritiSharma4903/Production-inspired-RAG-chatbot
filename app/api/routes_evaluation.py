import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db.models import EvaluationRun, EvaluationResult
from app.db.session import get_db
from app.evaluation.dataset_loader import (
    DATASET_DIR, DatasetExample, load_dataset, save_dataset, list_datasets,
)
from app.evaluation.evaluator import evaluate_single, run_dataset_evaluation
from app.evaluation.experiment import compare_runs, list_experiments
from app.evaluation.report_generator import build_report, print_console_report, save_json_report, save_csv_report
from app.evaluation.schemas import (
    EvaluationRequest, EvaluationResponse, DatasetCreateRequest,
    RunEvaluationRequest, RunEvaluationResponse, RunSummary,
)
from app.logging_config import get_logger

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
logger = get_logger(__name__)

REPORTS_DIR = os.environ.get("EVALUATION_REPORTS_DIR", "reports")


@router.post("/run", response_model=RunEvaluationResponse)
def run_evaluation(req: RunEvaluationRequest, db: Session = Depends(get_db)):
    """Batch-evaluates a named dataset file and persists an EvaluationRun.
    Runs synchronously (evaluation runs are bounded by dataset size, unlike
    document ingestion) -- for very large datasets, wrap this call in the
    same worker-queue pattern used for ingestion (app/workers/tasks.py)."""
    path = os.path.join(DATASET_DIR, req.dataset_name)
    try:
        examples = load_dataset(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset_name}")

    if not examples:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    run_id = run_dataset_evaluation(
        db, examples, req.experiment_name, req.dataset_name, req.document_id
    )
    run = db.get(EvaluationRun, run_id)
    return RunEvaluationResponse(run_id=run_id, status=run.status, total_questions=run.total_questions)


@router.post("/single", response_model=EvaluationResponse)
def evaluate_single_question(req: EvaluationRequest):
    """Scores ONE question immediately (no persistence) -- useful for
    ad-hoc debugging of a single bad answer without running a full dataset."""
    result = evaluate_single(req.question, req.ground_truth, req.document_id,
                              answer=req.answer, contexts=req.contexts)
    return EvaluationResponse(
        context_precision=result.get("context_precision"),
        context_recall=result.get("context_recall"),
        faithfulness=result.get("faithfulness"),
        answer_relevancy=result.get("answer_relevancy"),
        correctness=result.get("answer_correctness"),
        latency=result["latency_sec"],
        evaluator_used=result["evaluator_used"],
    )


@router.get("/results/{run_id}")
def get_run_results(run_id: str, db: Session = Depends(get_db)):
    results = db.query(EvaluationResult).filter(EvaluationResult.run_id == run_id).all()
    if not results:
        raise HTTPException(status_code=404, detail="Run not found or has no results")
    return [
        {
            "question": r.question, "answer": r.generated_answer, "ground_truth": r.ground_truth,
            "contexts": r.retrieved_contexts, "question_score": r.question_score, "passed": r.passed,
            "context_precision": r.context_precision, "context_recall": r.context_recall,
            "faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
        }
        for r in results
    ]


@router.get("/history", response_model=list[RunSummary])
def get_history(experiment_name: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    runs = list_experiments(db, experiment_name, limit)
    return [
        RunSummary(
            run_id=str(r.run_id), experiment_name=r.experiment_name, dataset_name=r.dataset_name,
            evaluator_used=r.evaluator_used, overall_score=r.overall_score, pass_count=r.pass_count,
            fail_count=r.fail_count, status=r.status, started_at=r.started_at, finished_at=r.finished_at,
        )
        for r in runs
    ]


@router.get("/report")
def get_report(run_id: str, format: str = "json", db: Session = Depends(get_db)):
    """format: json | csv | console (console prints server-side and also returns JSON)."""
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = db.query(EvaluationResult).filter(EvaluationResult.run_id == run_id).all()
    report = build_report(run, results)

    if format == "console":
        print_console_report(report)
        return report
    if format == "csv":
        path = save_csv_report(results, os.path.join(REPORTS_DIR, f"{run_id}.csv"))
        return {"report_path": path}
    if format == "json":
        path = save_json_report(report, os.path.join(REPORTS_DIR, f"{run_id}.json"))
        return report
    raise HTTPException(status_code=400, detail="format must be one of: json, csv, console")


@router.get("/compare")
def compare(run_ids: str, db: Session = Depends(get_db)):
    """run_ids: comma-separated list, e.g. ?run_ids=uuid1,uuid2"""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run_ids to compare")
    return compare_runs(db, ids)


@router.post("/dataset/create")
def create_dataset_endpoint(req: DatasetCreateRequest):
    examples = [
        DatasetExample(question=e.question, ground_truth=e.ground_truth,
                        expected_context=e.expected_context, metadata=e.metadata)
        for e in req.examples
    ]
    filename = req.dataset_name if req.dataset_name.endswith(".json") else f"{req.dataset_name}.json"
    path = save_dataset(examples, os.path.join(DATASET_DIR, filename))
    return {"status": "created", "path": path, "example_count": len(examples)}


@router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".json", ".csv")):
        raise HTTPException(status_code=400, detail="Only .json or .csv datasets are supported")
    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, file.filename)
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    examples = load_dataset(path)  # validate it parses before confirming success
    return {"status": "uploaded", "path": path, "example_count": len(examples)}


@router.get("/dataset/list")
def list_dataset_files():
    return {"datasets": list_datasets()}
