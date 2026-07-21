"""
report_generator.py
====================
Turns a persisted EvaluationRun + its EvaluationResults into the three
report formats the brief asks for: console (human-readable, for CLI use),
JSON (machine-readable, for the API/dashboard), CSV (spreadsheet-friendly,
for sharing with non-engineers).
"""

import csv
import json
import os

from app.db.models import EvaluationRun, EvaluationResult

METRIC_FIELDS = [
    "context_precision", "context_recall", "context_relevancy", "context_utilization",
    "faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity",
]


def build_report(run: EvaluationRun, results: list[EvaluationResult]) -> dict:
    sorted_results = sorted(results, key=lambda r: (r.question_score is None, r.question_score))
    lowest = sorted_results[:3]
    highest = sorted_results[-3:][::-1]

    return {
        "run_id": str(run.run_id),
        "experiment_name": run.experiment_name,
        "dataset_name": run.dataset_name,
        "evaluator_used": run.evaluator_used,
        "total_questions": run.total_questions,
        "execution_time_sec": run.execution_time_sec,
        "overall_score": run.overall_score,
        "pass_count": run.pass_count,
        "fail_count": run.fail_count,
        "pass_rate": run.pass_count / run.total_questions if run.total_questions else None,
        "metric_summary": {field: getattr(run, f"avg_{field}") for field in METRIC_FIELDS},
        "lowest_scoring_questions": [_result_summary(r) for r in lowest],
        "highest_scoring_questions": [_result_summary(r) for r in highest],
        "status": run.status,
        "error_message": run.error_message,
    }


def _result_summary(r: EvaluationResult) -> dict:
    return {
        "question": r.question,
        "answer": r.generated_answer,
        "question_score": r.question_score,
        "passed": r.passed,
    }


def print_console_report(report: dict) -> None:
    print("=" * 70)
    print(f"EVALUATION REPORT — {report['experiment_name']} ({report['dataset_name']})")
    print("=" * 70)
    print(f"Evaluator: {report['evaluator_used']}   Questions: {report['total_questions']}   "
          f"Time: {report['execution_time_sec']:.2f}s")
    print(f"Overall RAG Quality Score: {_fmt(report['overall_score'])}")
    print(f"Pass/Fail: {report['pass_count']} passed / {report['fail_count']} failed "
          f"({_fmt(report['pass_rate'], pct=True)})")
    print("-" * 70)
    print("Metric-wise averages:")
    for k, v in report["metric_summary"].items():
        print(f"  {k:<22} {_fmt(v)}")
    print("-" * 70)
    print("Lowest scoring questions:")
    for r in report["lowest_scoring_questions"]:
        print(f"  [{_fmt(r['question_score'])}] {r['question'][:80]}")
    print("Highest scoring questions:")
    for r in report["highest_scoring_questions"]:
        print(f"  [{_fmt(r['question_score'])}] {r['question'][:80]}")
    print("=" * 70)


def save_json_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def save_csv_report(results: list[EvaluationResult], path: str) -> str:
    """Per-question CSV -- one row per question, one column per metric.
    Distinct from the JSON report, which is run-level with only the
    top/bottom 3 questions inlined; the CSV has every question."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["question", "generated_answer", "ground_truth", "question_score", "passed"] + METRIC_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"question": r.question, "generated_answer": r.generated_answer,
                   "ground_truth": r.ground_truth, "question_score": r.question_score, "passed": r.passed}
            row.update({field: getattr(r, field) for field in METRIC_FIELDS})
            writer.writerow(row)
    return path


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"
