import hashlib
import os
from abc import ABC, abstractmethod
from app.config import settings

class FileStorage(ABC):
    @abstractmethod
    def save(self, document_id: str, version: int, filename: str, content: bytes) -> str:
        """Persist raw bytes, return a storage_path/key that read() can use later."""

    @abstractmethod
    def read(self, storage_path: str) -> bytes:
        """Return raw bytes for a previously saved file."""


class LocalFileStorage(FileStorage):
    """
    Stores files at:
        {FILE_STORAGE_ROOT}/{document_id}/v{version}__{filename}

    Versioning the path (not overwriting v1's file) is intentional: it lets
    rollback re-read an exact prior version's bytes without needing to
    re-download anything, and it keeps an audit trail of every uploaded
    version on disk.
    """

    def __init__(self, root: str | None = None):
        self.root = root or settings.FILE_STORAGE_ROOT
        os.makedirs(self.root, exist_ok=True)

    def _path_for(self, document_id: str, version: int, filename: str) -> str:
        doc_dir = os.path.join(self.root, document_id)
        os.makedirs(doc_dir, exist_ok=True)
        return os.path.join(doc_dir, f"v{version}__{filename}")

    def save(self, document_id: str, version: int, filename: str, content: bytes) -> str:
        path = self._path_for(document_id, version, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def read(self, storage_path: str) -> bytes:
        with open(storage_path, "rb") as f:
            return f.read()

def compute_file_hash(content: bytes) -> str:
    """
    Whole-file SHA-256. This is the CHEAPEST possible change-detection check
    in the entire pipeline: O(file_size) to compute, O(1) to compare against
    documents.latest_file_hash. If it matches, we skip parsing, chunking,
    diffing, embedding -- everything -- and return immediately. This is the
    first idempotency gate described in docs/UPDATE_PIPELINE.md.
    """
    return hashlib.sha256(content).hexdigest()


def get_file_storage() -> FileStorage:
    """Factory function -- the ONE place you'd change to switch to S3."""
    return LocalFileStorage()
