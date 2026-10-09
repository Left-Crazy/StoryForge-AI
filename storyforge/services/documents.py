from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .. import db
from ..config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from ..parser import normalize_text, parse_uploaded_file


def ingest_upload(project_id: str, filename: str, content_type: str, data: bytes) -> dict:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF, DOCX and TXT files are supported.")
    normalized, paragraphs = parse_uploaded_file(filename, data)
    original = normalized
    project_dir = UPLOAD_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    destination = project_dir / f"{uuid4().hex}{Path(filename).suffix.lower()}"
    destination.write_bytes(data)
    return db.save_document(project_id, filename, content_type or "", str(destination), original, normalized, paragraphs)


def ingest_text(project_id: str, title: str, text: str) -> dict:
    normalized, paragraphs = normalize_text(text)
    project_dir = UPLOAD_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    destination = project_dir / f"{uuid4().hex}.txt"
    destination.write_text(text, encoding="utf-8")
    return db.save_document(project_id, title or "pasted-story.txt", "text/plain", str(destination), text, normalized, paragraphs)


def latest(project_id: str) -> dict | None:
    return db.latest_document(project_id)
