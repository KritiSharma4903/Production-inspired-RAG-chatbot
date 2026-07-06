from dataclasses import dataclass
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from app.logging_config import get_logger
logger = get_logger(__name__)

@dataclass
class TextBlock:
    page_number: int
    order_index: int
    text: str

def parse_pdf(file_bytes: bytes) -> list[TextBlock]:
    """
    Extract text page by page from a PDF.
    """

    reader = PdfReader(BytesIO(file_bytes))

    blocks = []
    order_index = 0

    for page_num, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        blocks.append(
            TextBlock(
                page_number=page_num,
                order_index=order_index,
                text=text,
            )
        )
        order_index += 1
    logger.info(f"Parsed PDF into {len(blocks)} blocks")
    return blocks

def parse_docx(file_bytes: bytes) -> list[TextBlock]:
    """
    Extract text from DOCX.

    DOCX doesn't have page numbers,
    so page_number is always 1.
    """

    doc = Document(BytesIO(file_bytes))
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    blocks = []

    if full_text:
        blocks.append(
            TextBlock(
                page_number=1,
                order_index=0,
                text=full_text,
            )
        )
    logger.info(f"Parsed DOCX into {len(blocks)} block(s)")

    return blocks


def parse_document(filename: str, file_bytes: bytes) -> list[TextBlock]:
    """
    Automatically choose parser
    based on file extension.
    """

    filename = filename.lower()

    if filename.endswith(".pdf"):
        return parse_pdf(file_bytes)

    elif filename.endswith(".docx"):
        return parse_docx(file_bytes)

    else:
        raise ValueError(
            f"Unsupported file type: {filename}"
        )