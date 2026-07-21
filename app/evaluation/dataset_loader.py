"""
dataset_loader.py
==================
Loads/saves evaluation datasets in JSON or CSV. Each example has:
    question (required), ground_truth (optional), expected_context (optional
    list[str]), metadata (optional dict).

JSON is the canonical/richest format (supports nested expected_context and
metadata); CSV is supported for spreadsheet-authored datasets, with
expected_context stored as a `|`-delimited string and metadata as a flat
JSON string in a single column (documented limitation vs. JSON's native
nesting).
"""

import csv
import json
import os

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

DATASET_DIR = os.environ.get("EVALUATION_DATA_DIR", "evaluation_data")


class DatasetExample:
    def __init__(self, question: str, ground_truth: str | None = None,
                 expected_context: list[str] | None = None, metadata: dict | None = None):
        self.question = question
        self.ground_truth = ground_truth
        self.expected_context = expected_context or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_context": self.expected_context,
            "metadata": self.metadata,
        }


def load_dataset(path: str) -> list[DatasetExample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if path.endswith(".json"):
        return _load_json(path)
    if path.endswith(".csv"):
        return _load_csv(path)
    raise ValueError(f"Unsupported dataset format: {path} (expected .json or .csv)")


def _load_json(path: str) -> list[DatasetExample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = []
    for row in raw:
        examples.append(DatasetExample(
            question=row["question"],
            ground_truth=row.get("ground_truth"),
            expected_context=row.get("expected_context", []),
            metadata=row.get("metadata", {}),
        ))
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def _load_csv(path: str) -> list[DatasetExample]:
    examples = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_context = row.get("expected_context", "")
            metadata_raw = row.get("metadata", "")
            examples.append(DatasetExample(
                question=row["question"],
                ground_truth=row.get("ground_truth") or None,
                expected_context=[c.strip() for c in expected_context.split("|") if c.strip()],
                metadata=json.loads(metadata_raw) if metadata_raw else {},
            ))
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def save_dataset(examples: list[DatasetExample], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2)
    elif path.endswith(".csv"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "expected_context", "metadata"])
            writer.writeheader()
            for e in examples:
                writer.writerow({
                    "question": e.question,
                    "ground_truth": e.ground_truth or "",
                    "expected_context": "|".join(e.expected_context),
                    "metadata": json.dumps(e.metadata) if e.metadata else "",
                })
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    logger.info(f"Saved {len(examples)} examples to {path}")
    return path


def list_datasets() -> list[str]:
    """Used by GET-style dataset-listing endpoints and the Streamlit
    dashboard's dataset picker."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(f for f in os.listdir(DATASET_DIR) if f.endswith((".json", ".csv")))


