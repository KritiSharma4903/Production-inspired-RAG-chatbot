import re
from dataclasses import dataclass
from app.config import settings
from app.ingestion.hasher import content_hash, deterministic_chunk_id
from app.ingestion.parser import TextBlock

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page_number: int
    order_index: int
    text: str
    content_hash: str
    token_count: int
    char_start: int
    char_end: int


def _normalize(text: str) -> str:
    """Collapses whitespace variance so trivial re-encoding (e.g. different
    line-wrap) doesn't register as a content change. This normalization MUST
    be applied identically before hashing on every version, or hashes won't
    be comparable across versions."""
    return re.sub(r"\s+", " ", text).strip()


def _naive_tokenize(text: str) -> list[str]:
    """Whitespace tokenization stand-in. Swap for a real tokenizer
    (tiktoken / the embedding model's tokenizer) in production so chunk
    sizes are accurate against the model's actual context window."""
    return text.split(" ")


def chunk_text_block(block: TextBlock, document_id: str) -> list[Chunk]:
    tokens = _naive_tokenize(_normalize(block.text))
    size = settings.CHUNK_TOKEN_SIZE
    overlap = settings.CHUNK_OVERLAP_TOKENS
    stride = max(size - overlap, 1)

    chunks: list[Chunk] = []
    local_order = 0
    for start in range(0, len(tokens), stride):
        window = tokens[start: start + size]
        if not window:
            continue
        chunk_str = " ".join(window)
        c_hash = content_hash(chunk_str)
        chunk_id = deterministic_chunk_id(document_id, c_hash)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                page_number=block.page_number,
                order_index=block.order_index * 1000 + local_order,  
                text=chunk_str,
                content_hash=c_hash,
                token_count=len(window),
                char_start=start,
                char_end=start + len(window),
            )
        )
        local_order += 1
        if start + size >= len(tokens):
            break

    return chunks


def chunk_document(blocks: list[TextBlock], document_id: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for block in blocks:
        all_chunks.extend(chunk_text_block(block, document_id))
    return all_chunks
