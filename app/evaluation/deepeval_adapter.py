"""
deepeval_adapter.py
====================
NOT YET IMPLEMENTED -- this file exists to prove the adapter pattern
extends to DeepEval without refactoring evaluator.py. Every adapter
(metrics.py/native, ragas_metrics.py, this file) exposes the SAME shape:

    def evaluate_<adapter>(question, answer, contexts, ground_truth=None) -> dict

evaluator.py dispatches to whichever adapter is enabled via
settings.DEEPEVAL_ENABLED purely by function reference -- adding real
DeepEval support later means: (1) `pip install deepeval`, (2) fill in the
function body below using DeepEval's `GEval`/`FaithfulnessMetric` classes,
(3) flip DEEPEVAL_ENABLED=true in .env. No other file changes.
"""

from app.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_with_deepeval(
    question: str, answer: str, contexts: list[str], ground_truth: str | None = None
) -> dict:
    raise NotImplementedError(
        "DeepEval adapter is not implemented yet. Set DEEPEVAL_ENABLED=false "
        "in .env to use the native or Ragas evaluator instead. See this "
        "file's module docstring for the three-step plan to add it."
    )
