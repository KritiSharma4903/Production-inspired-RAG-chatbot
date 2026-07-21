"""
experiment.py
=============
Experiment comparison ON TOP OF the evaluation_runs table -- distinct from
langsmith_evaluator.py's compare_experiments(), which compares LangSmith's
own hosted experiment records. This module compares the project's own
Postgres-persisted runs, so comparison works even when LangSmith isn't
enabled (native/Ragas runs are still comparable to each other).
"""

from sqlalchemy.orm import Session

from app.db.models import EvaluationRun
from app.evaluation.report_generator import METRIC_FIELDS


def list_experiments(db: Session, experiment_name: str | None = None, limit: int = 50) -> list[EvaluationRun]:
    query = db.query(EvaluationRun).order_by(EvaluationRun.started_at.desc())
    if experiment_name:
        query = query.filter(EvaluationRun.experiment_name == experiment_name)
    return query.limit(limit).all()


def compare_runs(db: Session, run_ids: list[str]) -> dict:
    """
    Returns {"metric_name": {run_id: avg_score, ...}, ...} across the given
    runs, plus overall_score and pass_rate for a quick "which run is
    better" read. Use this to answer "did switching the chunk size /
    prompt template / embedding model improve quality?" by diffing two
    runs over the SAME dataset.
    """
    runs = db.query(EvaluationRun).filter(EvaluationRun.run_id.in_(run_ids)).all()
    comparison: dict[str, dict[str, float | None]] = {field: {} for field in METRIC_FIELDS}
    comparison["overall_score"] = {}
    comparison["pass_rate"] = {}

    for run in runs:
        label = f"{run.experiment_name} ({str(run.run_id)[:8]})"
        for field in METRIC_FIELDS:
            comparison[field][label] = getattr(run, f"avg_{field}")
        comparison["overall_score"][label] = run.overall_score
        comparison["pass_rate"][label] = run.pass_count / run.total_questions if run.total_questions else None

    return comparison
