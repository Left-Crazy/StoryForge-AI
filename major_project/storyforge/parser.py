from __future__ import annotations

from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from docx import Document as DocxDocument
from pypdf import PdfReader


def normalize_text(text: str) -> tuple[str, list[dict]]:
    lines = [" ".join(line.strip().split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    paragraphs = [{"id": f"p-{i}", "position": i, "text": line} for i, line in enumerate(lines, 1)]
    return "\n".join(lines), paragraphs


def parse_uploaded_file(filename: str, data: bytes) -> tuple[str, list[dict]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        text = data.decode("utf-8", errors="replace")
    elif suffix == ".docx":
        with NamedTemporaryFile(suffix=".docx", delete=True) as temp:
            temp.write(data); temp.flush()
            doc = DocxDocument(temp.name)
            text = "\n".join(p.text for p in doc.paragraphs)
    elif suffix == ".pdf":
        with NamedTemporaryFile(suffix=".pdf", delete=True) as temp:
            temp.write(data); temp.flush()
            reader = PdfReader(temp.name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError("Only PDF, DOCX and TXT files are supported.")
    return normalize_text(text)
