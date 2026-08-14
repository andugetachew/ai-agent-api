from __future__ import annotations
import io

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
}

# Caps how much extracted text gets stored/returned, so one huge upload can't
# blow out the DB row or flood an agent's context window if read via the tool.
MAX_EXTRACTED_CHARS = 50_000


class UnsupportedFileTypeError(Exception):
    pass


class ExtractionError(Exception):
    pass


def extract_text(filename: str, content_type: str, raw_bytes: bytes) -> str:
    """
    Extracts plain text from an uploaded file's raw bytes based on its
    content type. Raises UnsupportedFileTypeError for anything not in
    SUPPORTED_CONTENT_TYPES, and ExtractionError if parsing fails (e.g. a
    corrupt or password-protected PDF).
    """
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            f"unsupported content type '{content_type}' for file '{filename}'. "
            f"Supported: {sorted(SUPPORTED_CONTENT_TYPES)}"
        )

    try:
        if content_type == "application/pdf":
            text = _extract_pdf(raw_bytes)
        elif content_type == "text/plain":
            text = _extract_txt(raw_bytes)
        else:
            text = _extract_docx(raw_bytes)
    except UnsupportedFileTypeError:
        raise
    except Exception as exc:
        raise ExtractionError(f"failed to extract text from '{filename}': {exc}") from exc

    return text[:MAX_EXTRACTED_CHARS]


def _extract_pdf(raw_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text).strip()


def _extract_docx(raw_bytes: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs).strip()


def _extract_txt(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="replace").strip()