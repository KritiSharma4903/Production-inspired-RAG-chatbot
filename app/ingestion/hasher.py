import hashlib

def content_hash(text: str) -> str:
    """Full 256-bit hash of normalized chunk text. Used for the diff
    (old_hash != new_hash => chunk changed). O(len(text)) to compute,
    O(1) to compare."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_chunk_id(document_id: str, c_hash: str) -> str:
    """
    chunk_id is content-anchored, not position-anchored. This means:
      - Same paragraph, same document, different version -> SAME chunk_id.
      - Same paragraph moved to a different page -> SAME chunk_id (only
        its page_number/order_index metadata changes -- a metadata-only
        update, no re-embedding needed).
      - Even one character different -> DIFFERENT chunk_id (treated as a
        delete of the old id + insert of a new id, OR handled as an
        "update" by the diff engine depending on strategy -- see
        diff_engine.py for how we choose to treat this as an update when
        we can still positionally correlate old/new for a cleaner version
        history).
    """
    return f"{document_id}:{c_hash[:16]}"


def file_hash(content: bytes) -> str:
    """Whole-file hash -- see storage/file_storage.py for the actual
    implementation used at upload time (kept there so upload-time code has
    zero dependency on the ingestion package)."""
    return hashlib.sha256(content).hexdigest()
