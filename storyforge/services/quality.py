from __future__ import annotations

from uuid import uuid4

from .. import db
from ..ai.factory import get_provider
from .documents import latest
from .story import get as get_story_bible


def default_action(check_type: str, message: str = "") -> str:
    actions = {
        "character": "Check the character against the Story Bible. Rename the character to a known person, or add them to the Story Bible only if the source story supports them.",
        "location": "Confirm the scene location against the source story and Story Bible. Correct the slugline/location or update the Story Bible if the source supports the change.",
        "traceability": "Review the scene's source-event links and reconnect it to valid Story Bible event IDs. Do not remove a source beat just to silence the warning.",
        "generation": "Regenerate this scene, then confirm the screenplay editor contains a complete scene with a slugline and action.",
        "pacing": "Consider adding visual action beats, character reactions, or purposeful movement between dialogue exchanges. Keep only action that advances the scene.",
        "production": "Review the scene and add the detected production item to its production flags if it is genuinely required.",
        "scene_order": "Rebuild the scene plan or renumber scenes so numbering starts at 1 and increases without gaps.",
        "timeline": "Compare the event order with the source story and adjust scene order or time references where needed.",
        "plot_thread": "Check the referenced plot thread against the ending. Resolve it, preserve it intentionally as an open thread, or clarify the intended setup.",
        "unsupported_fact": "Verify this detail against the source. Remove it or rewrite it as an inference only if the story supports that inference.",
        "missing_beat": "Compare the planned scenes with the Story Bible's major events and add the missing beat if it is important to the source story.",
        "ai_review": "Read the review note alongside the source story and scene. Keep the warning only if you can confirm it; revise the scene or dismiss it if it is not a real issue.",
    }
    return actions.get(check_type, "Review the scene against the Story Bible and source story, then make a targeted edit if the issue is confirmed.")


def _issue(scene_id, check_type, severity, message, evidence="", suggested_action=None):
    return {
        "id": str(uuid4()),
        "scene_id": scene_id,
        "check_type": check_type,
        "severity": severity,
        "message": str(message or "A possible issue was detected; inspect the evidence below."),
        "evidence": str(evidence or ""),
        "suggested_action": str(suggested_action or default_action(check_type, str(message or ""))),
    }


def run(
    project_id: str,
    model_path: str | None = None,
    include_ai_review: bool = True,
) -> list[dict]:
    bible = get_story_bible(project_id)
    scenes = db.list_scenes(project_id)
    document = latest(project_id)
    if not bible:
        raise ValueError("Create the Story Bible before running quality checks.")
    if not scenes:
        raise ValueError("Generate at least one screenplay scene before running quality checks.")

    characters = {c.get("name") for c in bible.get("characters", []) if c.get("name")}
    locations = {str(loc).upper() for loc in bible.get("locations", []) if loc}
    event_ids = {e.get("id") for e in bible.get("major_events", []) if e.get("id")}
    issues = []

    for scene in scenes:
        for char in scene.get("characters", []):
            if characters and char not in characters:
                issues.append(_issue(
                    scene["id"], "character", "warning",
                    f"Scene uses character '{char}' who is not present in the Story Bible.",
                    evidence=f"Scene: {scene.get('slugline', 'Untitled scene')}; scene characters: {', '.join(scene.get('characters', []))}.",
                ))
        if locations and scene["location"].upper() not in locations and "UNSPECIFIED" not in locations:
            issues.append(_issue(
                scene["id"], "location", "warning",
                f"Scene location '{scene['location']}' is not listed in the Story Bible.",
                evidence=f"Scene slugline: {scene.get('slugline', 'Untitled scene')}; Story Bible locations: {', '.join(sorted(locations))}.",
            ))
        invalid_events = [eid for eid in scene.get("source_event_ids", []) if eid not in event_ids]
        if invalid_events:
            issues.append(_issue(
                scene["id"], "traceability", "warning",
                f"Scene references source event IDs that are not in the Story Bible: {', '.join(invalid_events)}.",
                evidence=f"Scene: {scene.get('slugline', 'Untitled scene')}; invalid IDs: {', '.join(invalid_events)}.",
            ))
        if not scene.get("screenplay_text", "").strip():
            issues.append(_issue(
                scene["id"], "generation", "error", "Scene exists in the plan but has no screenplay content.",
                evidence=f"Planned scene: {scene.get('slugline', 'Untitled scene')} — {scene.get('story_purpose', 'No purpose recorded')}.",
            ))
        dialogue_words = sum(len((d.get("text") or "").split()) for d in scene.get("dialogue", []))
        action_words = len((scene.get("action") or "").split())
        if dialogue_words and action_words < 12:
            issues.append(_issue(
                scene["id"], "pacing", "info", "Dialogue-heavy scene may benefit from more visual action.",
                evidence=f"Approximate action words: {action_words}; dialogue words: {dialogue_words}.",
            ))
        upper = scene.get("screenplay_text", "").upper()
        for marker in ["NIGHT", "VFX", "CROWD", "VEHICLE", "STUNT", "MUSIC"]:
            if marker in upper and marker.lower() not in [p.lower() for p in scene.get("production_flags", [])]:
                issues.append(_issue(
                    scene["id"], "production", "info", f"Production flag detected: {marker.lower()}.",
                    evidence=f"The text of '{scene.get('slugline', 'Untitled scene')}' includes the marker '{marker}'.",
                ))

    nums = sorted(s["scene_number"] for s in scenes)
    if nums != list(range(1, len(nums) + 1)):
        issues.append(_issue(None, "scene_order", "error", "Scene numbering is not contiguous.", evidence=f"Found scene numbers: {nums}."))

    # A real model can add semantic review on top of deterministic checks.
    provider = get_provider(model_path)
    if include_ai_review and provider.is_real_ai and document:
        try:
            ai_issues = provider.quality_review(bible, scenes, document["normalized_text"])
            by_num = {s["scene_number"]: s["id"] for s in scenes}
            for issue in ai_issues:
                severity = str(issue.get("severity", "warning")).lower()
                if severity not in {"error", "warning", "info"}:
                    severity = "warning"
                check_type = str(issue.get("check_type", "ai_review") or "ai_review")
                message = (
                    issue.get("message") or issue.get("finding") or issue.get("description")
                    or issue.get("problem") or issue.get("evidence")
                    or "The AI reviewer returned a finding without a detailed explanation. Inspect the suggested action and related scene."
                )
                try:
                    scene_number = int(issue.get("scene_number", 0) or 0)
                except (TypeError, ValueError):
                    scene_number = 0
                issues.append(_issue(
                    by_num.get(scene_number),
                    check_type,
                    severity,
                    message,
                    evidence=issue.get("evidence", issue.get("source_evidence", "")),
                    suggested_action=issue.get("suggested_action", issue.get("recommendation")),
                ))
        except Exception as exc:
            issues.append(_issue(None, "ai_review", "info", f"AI semantic review was unavailable: {exc}"))

    return db.replace_quality_issues(project_id, issues)
