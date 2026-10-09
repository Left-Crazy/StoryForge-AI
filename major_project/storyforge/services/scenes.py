from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from .. import db
from ..ai.factory import get_provider
from .documents import latest
from .story import get as get_story_bible


def _scene_source_context(story_bible: dict, scene: dict, document: dict) -> str:
    """Build a compact, scene-relevant source excerpt when event paragraph links exist.

    Story Bible event descriptions retain the overall plot; grounding a scene in its
    linked source paragraphs reduces repeated prompt tokens without changing the
    generation model, temperature, or screenplay instructions. If links are absent,
    preserve the previous full-source behavior.
    """
    full_text = str(document.get("normalized_text", "") or "")
    paragraphs = document.get("source_paragraphs") or []
    if not isinstance(paragraphs, list) or not paragraphs:
        return full_text

    paragraph_index: dict[str, int] = {}
    clean_paragraphs: list[dict] = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            continue
        paragraph_id = str(paragraph.get("id", "") or "")
        text = str(paragraph.get("text", "") or "").strip()
        if not text:
            continue
        clean_index = len(clean_paragraphs)
        clean_paragraphs.append({"id": paragraph_id or f"p-{index + 1}", "text": text})
        if paragraph_id:
            paragraph_index[paragraph_id] = clean_index
    if not clean_paragraphs:
        return full_text

    event_ids = {str(x) for x in scene.get("source_event_ids", [])}
    event_paragraph_ids: set[str] = set()
    events = story_bible.get("major_events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or str(event.get("id", "")) not in event_ids:
                continue
            source_ids = event.get("source_paragraph_ids", []) or event.get("source_paragraph_id", [])
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            if isinstance(source_ids, list):
                event_paragraph_ids.update(str(value) for value in source_ids if value)

    selected_indices: set[int] = set()
    for paragraph_id in event_paragraph_ids:
        index = paragraph_index.get(paragraph_id)
        if index is not None:
            # Include two neighbours on each side to preserve transitions and nuance.
            selected_indices.update(range(max(0, index - 2), min(len(clean_paragraphs), index + 3)))

    if not selected_indices:
        # Without precise source links, don't guess which excerpt matters.
        return full_text

    excerpts = [
        f"[{clean_paragraphs[index]['id']}] {clean_paragraphs[index]['text']}"
        for index in sorted(selected_indices)
    ]
    excerpt_text = "\n".join(excerpts)
    if len(full_text) <= 3000 or len(excerpt_text) < 200:
        return full_text
    return (
        "RELEVANT SOURCE EXCERPTS FOR THIS SCENE (Story Bible contains the overall plot; "
        "these linked paragraphs ground this particular scene):\n" + excerpt_text
    )


def plan(project_id: str, model_path: str | None = None) -> list[dict]:
    bible = get_story_bible(project_id)
    document = latest(project_id)
    if not bible or not document:
        raise ValueError("Create the Story Bible and keep the source story available before planning scenes.")

    provider = get_provider(model_path)
    planned = provider.plan_scenes(bible, document["normalized_text"])
    if not planned:
        raise ValueError("The AI provider returned an empty scene plan.")

    events = {e.get("id"): e for e in bible.get("major_events", [])}
    valid_characters = {c.get("name") for c in bible.get("characters", []) if c.get("name")}
    valid_locations = set(bible.get("locations", []))

    scenes: list[dict] = []
    for index, item in enumerate(planned, 1):
        scene_no = index
        characters = [c for c in item.get("characters", []) if c in valid_characters]
        if not characters:
            characters = list(valid_characters)[:2] or ["PROTAGONIST"]
        location = item.get("location") or (next(iter(valid_locations)) if valid_locations else "UNSPECIFIED")
        if valid_locations and location not in valid_locations:
            location = "UNSPECIFIED"
        slugline = item.get("slugline") or f"{item.get('interior_or_exterior', 'INT.').upper()} {location.upper()} - {item.get('time_of_day', 'DAY').upper()}"
        source_event_ids = [event_id for event_id in item.get("source_event_ids", []) if event_id in events]
        if not source_event_ids and bible.get("major_events"):
            source_event_ids = [bible["major_events"][min(index - 1, len(bible["major_events"]) - 1)]["id"]]

        scenes.append({
            "id": str(uuid4()),
            "scene_number": scene_no,
            "act": max(1, int(item.get("act", 1))),
            "sequence": max(1, int(item.get("sequence", 1))),
            "slugline": slugline.upper(),
            "interior_or_exterior": item.get("interior_or_exterior", "INT.").upper(),
            "location": location,
            "time_of_day": item.get("time_of_day", "UNSPECIFIED").upper(),
            "characters": characters,
            "story_purpose": item.get("story_purpose", "Advance the story.")[:1000],
            "source_event_ids": source_event_ids,
            "action": "",
            "dialogue": [],
            "emotional_tone": item.get("emotional_tone", "neutral"),
            "transitions": item.get("transitions", []),
            "props": item.get("props", []),
            "production_flags": item.get("production_flags", []),
            "estimated_duration": max(1, int(item.get("estimated_duration", 2))),
            "screenplay_text": "",
            "status": "planned",
        })
    return db.replace_scenes(project_id, scenes)


def list_all(project_id: str) -> list[dict]:
    return db.list_scenes(project_id)


def generate_one(project_id: str, scene_id: str, model_path: str | None = None) -> dict:
    bible = get_story_bible(project_id)
    document = latest(project_id)
    scene = next((s for s in db.list_scenes(project_id) if s["id"] == scene_id), None)
    if not bible or not document or not scene:
        raise ValueError("Story Bible, source story, and scene are required.")
    provider = get_provider(model_path)
    preceding = ""
    for previous in db.list_scenes(project_id):
        if previous["scene_number"] < scene["scene_number"]:
            preceding += previous.get("screenplay_text", "")[-2500:] + "\n\n"
    source_context = _scene_source_context(bible, scene, document)
    result = provider.generate_scene(bible, scene, source_context, preceding[-5000:])
    updated = db.update_scene(scene_id, result, status="generated")
    db.update_project_status(project_id, "generated")
    return updated


def generate_all(
    project_id: str,
    model_path: str | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> list[dict]:
    bible = get_story_bible(project_id)
    document = latest(project_id)
    if not bible or not document:
        raise ValueError("Create the Story Bible and scene plan before generating the screenplay.")
    provider = get_provider(model_path)
    scenes = db.list_scenes(project_id)
    if not scenes:
        raise ValueError("Build the scene plan before generating the screenplay.")

    preceding = ""
    total = len(scenes)
    for index, scene in enumerate(scenes, 1):
        if progress_callback:
            progress_callback(
                "generating", index - 1, total,
                f"Generating scene {index} of {total}: {scene.get('slugline', 'Untitled scene')}…",
            )
        source_context = _scene_source_context(bible, scene, document)
        result = provider.generate_scene(bible, scene, source_context, preceding[-5000:])
        db.update_scene(scene["id"], result, status="generated")
        preceding = (preceding + "\n\n" + result.get("screenplay_text", ""))[-7000:]
        if progress_callback:
            progress_callback(
                "generating", index, total,
                f"Completed scene {index} of {total}: {scene.get('slugline', 'Untitled scene')}",
            )
    db.update_project_status(project_id, "generated")
    return db.list_scenes(project_id)


def build_plan_and_generate(
    project_id: str,
    model_path: str | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> list[dict]:
    if progress_callback:
        progress_callback("planning", 0, 0, "Planning scene objectives and source links…")
    planned = plan(project_id, model_path)
    if progress_callback:
        progress_callback("planned", 0, len(planned), f"Scene plan ready: {len(planned)} scenes. Generating screenplay…")
    return generate_all(project_id, model_path, progress_callback=progress_callback)


def revise(project_id: str, scene_id: str, instruction: str, model_path: str | None = None) -> dict:
    if not instruction.strip():
        raise ValueError("Enter an instruction for the targeted revision.")
    bible = get_story_bible(project_id)
    document = latest(project_id)
    scene = next((s for s in db.list_scenes(project_id) if s["id"] == scene_id), None)
    if not bible or not document or not scene:
        raise ValueError("Story Bible, source story, and scene are required.")
    result = get_provider(model_path).revise_scene(bible, scene, instruction.strip(), document["normalized_text"])
    return db.update_scene(scene_id, result, status="edited")


def save_edits(scene_id: str, action: str, screenplay_text: str) -> dict:
    if not screenplay_text.strip():
        raise ValueError("The screenplay editor cannot be saved empty. Generate the scene first.")
    return db.update_scene(scene_id, {"action": action, "screenplay_text": screenplay_text}, status="edited")
