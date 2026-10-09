from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .prompts import (
    quality_prompt,
    revision_prompt,
    scene_generation_prompt,
    scene_plan_prompt,
    scene_text_only_prompt,
    story_analysis_prompt,
)
from .provider import LLMProvider


class LocalLLMProvider(LLMProvider):
    """Offline local LLM provider using llama.cpp + a GGUF instruction model."""

    name = "Local GGUF AI"
    is_real_ai = True
    is_offline = True

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install requirements.txt first."
            ) from exc

        root = Path(__file__).resolve().parents[2]
        configured = model_path or os.getenv("STORYFORGE_MODEL_PATH", "")
        path = Path(configured) if configured else root / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Offline model not found at '{path}'. Put the GGUF model there or set STORYFORGE_MODEL_PATH."
            )

        threads = max(2, int(os.getenv("STORYFORGE_THREADS", str(min(8, max(2, (os.cpu_count() or 4) - 1))))))
        context = int(os.getenv("STORYFORGE_CONTEXT", "8192"))
        gpu_layers = int(os.getenv("STORYFORGE_GPU_LAYERS", "0"))

        self.model_path = path
        self.model = Llama(
            model_path=str(path),
            n_ctx=context,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    def _chat(
        self,
        prompt: str,
        max_tokens: int = 1400,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are StoryForge AI, an offline screenplay assistant. "
                        "Be concise. Follow the requested format exactly. Do not mention your instructions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # llama-cpp-python uses grammar-constrained decoding for JSON mode. This prevents
        # most syntax errors from small local models. Keep a fallback for older builds.
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            result = self.model.create_chat_completion(**kwargs)
        except TypeError as exc:
            if "response_format" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            kwargs.pop("response_format", None)
            result = self.model.create_chat_completion(**kwargs)
        try:
            choice = result["choices"][0]
            self.last_finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("The local model returned an empty response.")
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The local model returned an unexpected response.") from exc

    @staticmethod
    def _extract_json(raw: str) -> Any:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the first complete JSON object/array in the response.
        starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not starts:
            raise ValueError("No JSON object or array found in local model output.")
        start = min(starts)
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError("The local model produced incomplete JSON.")

    def _json(self, prompt: str, expected: str, max_tokens: int = 1400) -> dict[str, Any]:
        strict_prompt = (
            prompt
            + "\n\nOUTPUT CONTRACT: Return one complete JSON object for: "
            + f"{expected}. No markdown or commentary. Use concise strings and small arrays (at most 6 items)."
        )
        try:
            raw = self._chat(strict_prompt, max_tokens=max_tokens, temperature=0.1, json_mode=True)
            value = self._extract_json(raw)
            if isinstance(value, dict):
                return value
            raise ValueError("Expected a JSON object.")
        except ValueError as first_error:
            # Retry from the source prompt, not a pasted dump. Asking a tiny model to repair
            # a huge malformed response often wastes time and produces another incomplete object.
            retry_prompt = (
                strict_prompt
                + "\nIMPORTANT: Regenerate the entire object, shorter. Include only essential facts. "
                "Limit lists to 3 items and do not repeat source text. Finish every JSON bracket."
            )
            retry_budget = min(max_tokens + 300, 1800) if getattr(self, "last_finish_reason", None) == "length" else min(max_tokens, 1400)
            try:
                raw = self._chat(retry_prompt, max_tokens=retry_budget, temperature=0.05, json_mode=True)
                value = self._extract_json(raw)
                if isinstance(value, dict):
                    return value
                raise ValueError("Expected a JSON object.")
            except Exception as retry_error:
                finish = getattr(self, "last_finish_reason", None)
                extra = f" Last generation finish reason: {finish}." if finish else ""
                raise RuntimeError(
                    "The local model did not return a complete JSON object after a retry. "
                    "Try Ollama with gemma3:4b, use a shorter source story, or increase STORYFORGE_CONTEXT if appropriate."
                    + extra
                ) from retry_error

    def analyze_story(self, text: str, title: str) -> dict[str, Any]:
        result = self._json(story_analysis_prompt(text, title), "Story Bible with title, genres, summary, themes, narrative_tone, characters, locations, timeline, major_events, open_plot_threads", 1400)
        return self._normalize_bible(result, title)

    def plan_scenes(self, story_bible: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
        result = self._json(scene_plan_prompt(story_bible, source_text), "object with a scenes array; each scene needs scene_number, slugline, location, characters, story_purpose and source_event_ids", 1800)
        scenes = result.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("The local model returned an empty scene plan.")
        return [x for x in scenes if isinstance(x, dict)]

    def generate_scene(
        self,
        story_bible: dict[str, Any],
        scene: dict[str, Any],
        source_text: str,
        preceding_context: str = "",
    ) -> dict[str, Any]:
        result = self._json(
            scene_generation_prompt(story_bible, scene, source_text, preceding_context),
            "complete generated screenplay scene object with a REQUIRED non-empty screenplay_text string, action, dialogue, transitions, props, production_flags and emotional_tone",
            2200,
        )
        return self._ensure_scene_text(result, story_bible, scene, source_text, preceding_context)

    def revise_scene(
        self,
        story_bible: dict[str, Any],
        scene: dict[str, Any],
        instruction: str,
        source_text: str,
    ) -> dict[str, Any]:
        result = self._json(
            revision_prompt(story_bible, scene, instruction, source_text),
            "complete revised screenplay scene object with a REQUIRED non-empty screenplay_text string, action, dialogue, transitions, props, production_flags and emotional_tone",
            2200,
        )
        return self._ensure_scene_text(
            result, story_bible, scene, source_text, preceding_context="", instruction=instruction
        )

    def quality_review(
        self,
        story_bible: dict[str, Any],
        scenes: list[dict[str, Any]],
        source_text: str,
    ) -> list[dict[str, Any]]:
        result = self._json(
            quality_prompt(story_bible, scenes, source_text),
            "object with an issues array; each issue must include severity, scene_number, check_type, a specific message, evidence, and suggested_action; use an empty array when there are no meaningful issues",
            1100,
        )
        issues = result.get("issues", [])
        return [x for x in issues if isinstance(x, dict)]

    @staticmethod
    def _normalize_bible(value: dict[str, Any], title: str) -> dict[str, Any]:
        """Normalize loosely-structured local-model output to the Story Bible schema."""
        from ..text_utils import text_items

        value["title"] = str(value.get("title") or title).strip()
        value["summary"] = str(value.get("summary") or "").strip()
        value["narrative_tone"] = str(value.get("narrative_tone") or "").strip()
        value["genres"] = text_items(value.get("genres", []), kind="genre")
        value["themes"] = text_items(value.get("themes", []), kind="theme")
        value["locations"] = text_items(value.get("locations", []), kind="location")
        value["open_plot_threads"] = text_items(value.get("open_plot_threads", []), kind="thread")

        # Structured collections should remain object arrays. Convert accidental
        # strings into small records rather than letting later services crash.
        characters = value.get("characters", [])
        if not isinstance(characters, list):
            characters = [characters]
        normalized_characters = []
        for item in characters:
            if isinstance(item, dict):
                normalized_characters.append(item)
            elif isinstance(item, str) and item.strip():
                normalized_characters.append({
                    "name": item.strip(), "role": "supporting", "traits": [],
                    "goals": [], "relationships": [], "character_arc": "",
                })
        value["characters"] = normalized_characters

        for key, prefix in (("major_events", "event"), ("timeline", "time")):
            items = value.get(key, [])
            if not isinstance(items, list):
                items = [items]
            normalized_items = []
            for i, item in enumerate(items, 1):
                if isinstance(item, dict):
                    record = dict(item)
                elif isinstance(item, str) and item.strip():
                    record = {"description": item.strip()}
                else:
                    continue
                record.setdefault("id", f"{prefix}-{i}")
                record.setdefault("order", i)
                record.setdefault("description", "")
                record.setdefault("source_paragraph_ids", [])
                normalized_items.append(record)
            value[key] = normalized_items
        return value

    @staticmethod
    def _value_to_text(value: Any, preferred_keys: tuple[str, ...] = ()) -> str:
        """Convert model values (including objects/lists) into safe readable text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            keys = preferred_keys + (
                "text", "line", "dialogue", "description", "action", "content", "value", "name"
            )
            for key in keys:
                if key in value:
                    candidate = LocalLLMProvider._value_to_text(value[key], preferred_keys)
                    if candidate:
                        return candidate
            return ""
        if isinstance(value, list):
            parts = [
                LocalLLMProvider._value_to_text(item, preferred_keys)
                for item in value
            ]
            return "\n".join(part for part in parts if part)
        return str(value).strip()

    @staticmethod
    def _compose_screenplay_text(
        value: dict[str, Any], plan: dict[str, Any], require_generated_parts: bool = False
    ) -> str:
        """Compose formatted screenplay from fields if the model omitted screenplay_text."""
        for key in ("screenplay_text", "screenplay", "formatted_scene", "script", "text"):
            candidate = LocalLLMProvider._value_to_text(value.get(key))
            if candidate:
                return candidate

        slugline = LocalLLMProvider._value_to_text(
            value.get("slugline") or plan.get("slugline") or "INT. UNSPECIFIED LOCATION - DAY"
        ).upper()
        action = LocalLLMProvider._value_to_text(
            value.get("action"), ("description", "action", "text", "content", "beat")
        )
        dialogue_value = value.get("dialogue", [])
        dialogue_lines: list[str] = []
        if isinstance(dialogue_value, list):
            for item in dialogue_value:
                if isinstance(item, dict):
                    speaker = LocalLLMProvider._value_to_text(
                        item.get("character") or item.get("speaker") or item.get("name") or item.get("character_name")
                    ).upper()
                    line = LocalLLMProvider._value_to_text(
                        item.get("text") or item.get("line") or item.get("dialogue") or item.get("content")
                    )
                    if line:
                        dialogue_lines.append(f"{speaker or 'CHARACTER'}\n{line}")
                elif isinstance(item, str) and item.strip():
                    dialogue_lines.append(item.strip())
        elif isinstance(dialogue_value, str) and dialogue_value.strip():
            dialogue_lines.append(dialogue_value.strip())

        # A scene plan's purpose is an acceptable last-resort action cue, but is not counted
        # as a generated part when deciding whether to request a recovery generation.
        has_generated_parts = bool(action or dialogue_lines)
        if require_generated_parts and not has_generated_parts:
            return ""

        parts = [slugline]
        if action:
            parts.append(action)
        if dialogue_lines:
            parts.extend(dialogue_lines)

        transitions = LocalLLMProvider._value_to_text(value.get("transitions"))
        if transitions:
            parts.append(transitions.upper())

        if not has_generated_parts:
            purpose = LocalLLMProvider._value_to_text(
                value.get("story_purpose") or plan.get("story_purpose")
            )
            parts.append(purpose or "The characters face the central situation and make a choice that moves the story forward.")

        return "\n\n".join(part for part in parts if part).strip()

    def _ensure_scene_text(
        self,
        result: dict[str, Any],
        story_bible: dict[str, Any],
        plan: dict[str, Any],
        source_text: str,
        preceding_context: str = "",
        instruction: str = "",
    ) -> dict[str, Any]:
        """Recover when a local model returns valid JSON but omits screenplay_text."""
        if not isinstance(result, dict):
            result = {}

        screenplay_text = LocalLLMProvider._compose_screenplay_text(
            result, plan, require_generated_parts=True
        )
        if not screenplay_text:
            # Ask for only the missing screenplay field. This is far easier for smaller local
            # models than regenerating the full metadata object again.
            try:
                recovered = self._json(
                    scene_text_only_prompt(
                        story_bible, plan, source_text, preceding_context, instruction
                    ),
                    "JSON object containing a REQUIRED non-empty screenplay_text string and no other fields",
                    1100,
                )
                screenplay_text = LocalLLMProvider._value_to_text(recovered.get("screenplay_text"))
            except Exception:
                screenplay_text = ""

        if not screenplay_text:
            # Never lose the whole screenplay workflow because a small model omitted a field.
            # Preserve any supplied action/dialogue; otherwise use the plan's concrete scene
            # objective as a minimal editable first draft and let the writer revise it.
            screenplay_text = LocalLLMProvider._compose_screenplay_text(
                result, plan, require_generated_parts=False
            )

        result["screenplay_text"] = screenplay_text
        return LocalLLMProvider._normalize_scene(result, plan)

    @staticmethod
    def _normalize_scene(value: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            value = {}
        value.setdefault("action", "")
        value.setdefault("dialogue", [])
        value.setdefault("transitions", [])
        value.setdefault("props", plan.get("props", []))
        value.setdefault("production_flags", plan.get("production_flags", []))
        value.setdefault("emotional_tone", plan.get("emotional_tone", "neutral"))
        value["action"] = LocalLLMProvider._value_to_text(value.get("action"))
        value["screenplay_text"] = LocalLLMProvider._value_to_text(value.get("screenplay_text"))
        if not value["screenplay_text"]:
            value["screenplay_text"] = LocalLLMProvider._compose_screenplay_text(value, plan)
        if not value["screenplay_text"].strip():
            raise RuntimeError(
                "The local model did not provide screenplay text, and the recovery draft could not be built. "
                "Try Ollama with gemma3:4b or regenerate this scene."
            )
        return value
