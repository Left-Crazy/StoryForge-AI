from __future__ import annotations

import re
from typing import Any

from .provider import LLMProvider

GENRE_KEYWORDS = {
    "thriller": ["threat", "murder", "chase", "secret", "danger", "killer", "attack"],
    "romance": ["love", "kiss", "relationship", "heart", "romantic"],
    "drama": ["family", "grief", "conflict", "father", "mother", "daughter", "son", "loss"],
    "comedy": ["joke", "funny", "laugh", "awkward", "comic"],
    "mystery": ["clue", "mystery", "investigate", "evidence", "detective", "unknown"],
    "fantasy": ["magic", "dragon", "kingdom", "spell", "wizard"],
    "science fiction": ["space", "robot", "planet", "spaceship", "android"],
}


class HeuristicProvider(LLMProvider):
    """Offline fallback. It keeps the demo usable but is explicitly not the real AI path."""

    name = "Offline demo provider"
    is_real_ai = False

    def analyze_story(self, text: str, title: str) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lower = text.lower()
        genres = [genre.title() for genre, words in GENRE_KEYWORDS.items() if any(word in lower for word in words)] or ["Drama"]
        names = self._extract_names(text)
        characters = [
            {"name": name, "role": "protagonist" if i == 0 else "supporting", "traits": [], "goals": [], "relationships": [], "character_arc": ""}
            for i, name in enumerate(names[:8])
        ] or [{"name": "Protagonist", "role": "protagonist", "traits": [], "goals": [], "relationships": [], "character_arc": ""}]
        events = self._extract_events(lines)
        return {
            "title": title,
            "genres": genres[:3],
            "summary": " ".join(lines)[:1200],
            "themes": self._themes(lower),
            "narrative_tone": "tense" if any(w in lower for w in ["danger", "threat", "chase"]) else "dramatic",
            "characters": characters,
            "locations": self._extract_locations(text) or ["UNSPECIFIED"],
            "timeline": [{**event} for event in events],
            "major_events": events,
            "open_plot_threads": self._threads(lower),
        }

    def plan_scenes(self, story_bible: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
        events = story_bible.get("major_events", [])
        chars = [c.get("name") for c in story_bible.get("characters", []) if c.get("name")]
        locations = story_bible.get("locations") or ["UNSPECIFIED"]
        count = min(max(3, len(events)), 8) if len(events) <= 4 else min(len(events), 10)
        scenes = []
        selected_events = (events * count)[:count] if events else [{"id": "event-1", "description": "Story begins."}]
        for i in range(count):
            event = selected_events[i]
            location = locations[i % len(locations)]
            act = 1 if i < count / 3 else 2 if i < (2 * count / 3) else 3
            scenes.append({
                "scene_number": i + 1,
                "act": act,
                "sequence": ((i % 3) + 1),
                "slugline": f"INT. {str(location).upper()} - DAY",
                "interior_or_exterior": "INT.",
                "location": location,
                "time_of_day": "DAY",
                "characters": chars[:2],
                "story_purpose": event.get("description", "Advance the story."),
                "source_event_ids": [event.get("id", f"event-{i+1}")],
                "emotional_tone": "rising tension" if act >= 2 else "setup",
                "transitions": [],
                "props": [],
                "production_flags": [],
                "estimated_duration": 2 if i < count - 1 else 3,
            })
        return scenes

    def generate_scene(self, story_bible: dict[str, Any], scene: dict[str, Any], source_text: str, preceding_context: str = "") -> dict[str, Any]:
        chars = scene.get("characters") or ["PROTAGONIST"]
        lead = chars[0]
        other = chars[1] if len(chars) > 1 else None
        objective = scene.get("story_purpose", "Advance the story").rstrip(".")
        action = f"{lead} enters {scene.get('location', 'the location').lower()} and confronts the immediate objective: {objective}."
        if other:
            action += f" {other} watches closely, raising the pressure."
        dialogue = [{"character": lead.upper(), "parenthetical": "", "text": "We need to deal with this now."}]
        if other:
            dialogue.append({"character": other.upper(), "parenthetical": "", "text": "Then tell me what you know."})
        screenplay = self._format(scene, action, dialogue)
        return {"action": action, "dialogue": dialogue, "transitions": [], "props": [], "production_flags": [], "emotional_tone": scene.get("emotional_tone", "neutral"), "screenplay_text": screenplay, "status": "generated"}

    def revise_scene(self, story_bible: dict[str, Any], scene: dict[str, Any], instruction: str, source_text: str) -> dict[str, Any]:
        result = self.generate_scene(story_bible, scene, source_text)
        text = instruction.lower()
        if "tension" in text or "darker" in text:
            result["emotional_tone"] = "tense"
            result["action"] += " A beat of silence makes the danger feel closer."
            result["dialogue"][0]["text"] = "We may not get another chance."
        if "natural" in text or "dialogue" in text:
            result["dialogue"][0]["text"] = "We need to handle this before it gets worse."
        if "short" in text:
            result["action"] = result["action"].split(".")[0] + "."
            result["dialogue"] = result["dialogue"][:1]
        result["screenplay_text"] = self._format(scene, result["action"], result["dialogue"])
        return result

    def _extract_names(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in [
            r"\b([A-Z][a-z]{2,})(?:,)?\s+(?:is|was|finds|find|walks|runs|enters|says|sees|opens)\b",
            r"\b([A-Z][a-z]{2,})\s*:\s*",
            r"\b([A-Z][a-z]{2,})\s*,\s*(?:a|an|the)\s+",
        ]:
            for match in re.findall(pattern, text):
                if match not in found and match not in {"The", "Then", "When", "After", "Before"}:
                    found.append(match)
        return found

    def _extract_locations(self, text: str) -> list[str]:
        out: list[str] = []
        for match in re.findall(r"\b(?:at|inside|outside|near|in)\s+(?:the\s+)?([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})", text):
            clean = match.strip(".,")
            if len(clean) > 2 and clean not in out:
                out.append(clean)
        return out[:8]

    def _extract_events(self, lines: list[str]) -> list[dict[str, Any]]:
        if not lines:
            return [{"id": "event-1", "order": 1, "description": "Story begins.", "source_paragraph_ids": []}]
        return [
            {"id": f"event-{i}", "order": i, "description": line[:300], "source_paragraph_ids": [f"p-{i}"]}
            for i, line in enumerate(lines[:12], 1)
        ]

    def _themes(self, text: str) -> list[str]:
        mapping = {
            "trust": ["trust", "betray"],
            "survival": ["survive", "danger", "escape"],
            "family": ["family", "father", "mother", "daughter", "son"],
            "identity": ["identity", "secret"],
        }
        return [theme for theme, words in mapping.items() if any(word in text for word in words)][:5] or ["change"]

    def _threads(self, text: str) -> list[str]:
        threads: list[str] = []
        if "secret" in text:
            threads.append("A secret is introduced and should be tracked across scenes.")
        if any(word in text for word in ["mystery", "clue", "unknown"]):
            threads.append("The central mystery should be resolved or intentionally left open.")
        return threads

    def _format(self, scene: dict[str, Any], action: str, dialogue: list[dict[str, str]]) -> str:
        lines = [scene.get("slugline", "INT. UNSPECIFIED - DAY").upper(), "", action, ""]
        for item in dialogue:
            lines.append(item["character"])
            if item.get("parenthetical"):
                lines.append(f"({item['parenthetical']})")
            lines.extend([item["text"], ""])
        return "\n".join(lines).strip()
