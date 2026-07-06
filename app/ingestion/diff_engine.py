from dataclasses import dataclass, field
from app.ingestion.chunker import Chunk

@dataclass
class DiffResult:
    unchanged: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    inserted: list[str] = field(default_factory=list)

    @property
    def to_embed(self) -> list[str]:
        """Chunks requiring a NEW embedding call: changed + inserted only.
        This property is the literal cost-saving mechanism -- everything
        downstream that touches the embedding API reads from here, never
        from the full chunk set."""
        return self.changed + self.inserted


def diff_chunk_sets(old_hashes: dict[str, str], new_hashes: dict[str, str]) -> DiffResult:
    old_ids = set(old_hashes.keys())
    new_ids = set(new_hashes.keys())

    unchanged, changed = [], []
    for cid in old_ids & new_ids:
        if old_hashes[cid] == new_hashes[cid]:
            unchanged.append(cid)
        else:
            changed.append(cid)

    deleted = list(old_ids - new_ids)
    inserted = list(new_ids - old_ids)

    return DiffResult(unchanged=unchanged, changed=changed, deleted=deleted, inserted=inserted)


def build_hash_map(chunks: list[Chunk]) -> dict[str, str]:
    """Small helper -- new_hashes side of the diff comes from freshly
    parsed+chunked Chunk objects; old_hashes side comes from a DB query
    (see pipeline.py) shaped into the same {chunk_id: content_hash} form."""
    return {c.chunk_id: c.content_hash for c in chunks}
