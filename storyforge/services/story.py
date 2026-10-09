from __future__ import annotations

from .. import db
from ..ai.factory import get_provider
from ..ai.local_provider import LocalLLMProvider
from .documents import latest


def analyze(project_id: str, model_path: str | None = None) -> dict:
    document = latest(project_id)
    if not document:
        raise ValueError("Upload or paste a story before analysis.")
    project = db.get_project(project_id)
    provider = get_provider(model_path)
    bible = provider.analyze_story(document["normalized_text"], project["title"])
    bible["title"] = bible.get("title") or project["title"]
    return db.upsert_story_bible(project_id, bible, status="analyzed")


def get(project_id: str) -> dict | None:
    bible = db.get_story_bible(project_id)
    if not bible:
        return None
    # Old runs may have persisted locations/genres/themes as objects. Normalize
    # them on read so existing projects are safe without requiring re-analysis.
    return LocalLLMProvider._normalize_bible(dict(bible), bible.get("title", "Untitled Story"))


def update(project_id: str, bible: dict) -> dict:
    return db.upsert_story_bible(project_id, bible, status="bible_reviewed")
