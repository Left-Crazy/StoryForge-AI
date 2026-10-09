from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .prompts import (
    quality_prompt,
    revision_prompt,
    scene_generation_prompt,
    scene_plan_prompt,
    story_analysis_prompt,
)
from .provider import LLMProvider
from .local_provider import LocalLLMProvider


class OllamaProvider(LLMProvider):
    """Offline provider that calls a locally running Ollama server over localhost."""

    name = "Ollama Local AI"
    is_real_ai = True
    is_offline = True

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        self.model_name = (model_name or os.getenv("STORYFORGE_OLLAMA_MODEL", "gemma3:4b")).strip()
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        if not self.model_name:
            raise ValueError("Enter an Ollama model name, for example gemma3:4b.")
        self.timeout = float(os.getenv("STORYFORGE_OLLAMA_TIMEOUT", "600"))
        self.num_ctx = int(os.getenv("STORYFORGE_CONTEXT", "8192"))
        self.last_eval_tokens: int | None = None
        self.last_duration_seconds: float | None = None

    def installed_models(self) -> list[str]:
        """Return locally installed model names without loading one into memory."""
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Cannot connect to Ollama at http://127.0.0.1:11434. Open the Ollama app, "
                "then try again. This connection stays on your own computer."
            ) from exc
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]

    def check_model(self) -> tuple[bool, str]:
        try:
            models = self.installed_models()
        except RuntimeError as exc:
            return False, str(exc)
        if self.model_name not in models:
            names = ", ".join(models) if models else "(no models found)"
            return False, f"Model '{self.model_name}' is not installed in Ollama. Installed models: {names}"
        return True, f"Ollama ready · {self.model_name} · local/offline inference"

    def _chat(self, prompt: str, max_tokens: int = 1400, temperature: float = 0.15) -> str:
        compact_system = (
            "You are StoryForge AI. Return one complete JSON object only. "
            "Use concise values and small arrays. Do not use markdown or commentary."
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": compact_system},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Ollama did not complete the local generation. Make sure Ollama is running and "
                "the selected model is installed. No internet/API key is used by this provider."
            ) from exc

        message = result.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response. Try a larger local model such as gemma3:4b.")
        eval_count = result.get("eval_count")
        total_duration = result.get("total_duration")
        self.last_eval_tokens = int(eval_count) if isinstance(eval_count, (int, float)) else None
        self.last_duration_seconds = (
            float(total_duration) / 1_000_000_000 if isinstance(total_duration, (int, float)) else None
        )
        return content.strip()

    def _json(self, prompt: str, expected: str, max_tokens: int = 1400) -> dict[str, Any]:
        contract = (
            "\n\nOUTPUT CONTRACT: Return one complete JSON object matching the requested task: "
            f"{expected}. Use short strings, no markdown, and no commentary. "
            "Keep arrays concise (normally no more than 6 items). Do not include fields not requested."
        )
        first_prompt = prompt + contract
        try:
            value = LocalLLMProvider._extract_json(self._chat(first_prompt, max_tokens=max_tokens))
            if isinstance(value, dict):
                return value
            raise ValueError("Expected a JSON object.")
        except ValueError as first_error:
            # A compact retry is cheaper and more useful than asking a small model to repair a long dump.
            retry_prompt = (
                prompt
                + contract
                + "\nIMPORTANT: The previous response was incomplete. Regenerate a shorter complete object. "
                "Include only essential facts and limit every list to at most 3 items."
            )
            try:
                value = LocalLLMProvider._extract_json(
                    self._chat(retry_prompt, max_tokens=max(1100, min(max_tokens, 1600)), temperature=0.05)
                )
            except Exception as retry_error:
                raise RuntimeError(
                    "The local model did not return complete JSON after a retry. Try the installed "
                    "gemma3:4b model instead of qwen2.5:0.5b, or use a shorter source story."
                ) from retry_error
            if not isinstance(value, dict):
                raise RuntimeError("Ollama returned JSON, but it was not a JSON object.") from first_error
            return value

    def analyze_story(self, text: str, title: str) -> dict[str, Any]:
        result = self._json(
            story_analysis_prompt(text, title),
            "Story Bible with title, genres, summary, themes, narrative_tone, characters, locations, timeline, major_events, open_plot_threads",
            1400,
        )
        return LocalLLMProvider._normalize_bible(result, title)

    def plan_scenes(self, story_bible: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
        result = self._json(
            scene_plan_prompt(story_bible, source_text),
            "object with a scenes array; each scene needs scene_number, slugline, location, characters, story_purpose and source_event_ids",
            1800,
        )
        scenes = result.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("The local Ollama model returned an empty scene plan. Try gemma3:4b.")
        return [item for item in scenes if isinstance(item, dict)]

    def generate_scene(
        self,
        story_bible: dict[str, Any],
        scene: dict[str, Any],
        source_text: str,
        preceding_context: str = "",
    ) -> dict[str, Any]:
        result = self._json(
            scene_generation_prompt(story_bible, scene, source_text, preceding_context),
            "complete screenplay scene object with a REQUIRED non-empty screenplay_text string, action, dialogue, transitions, props, production_flags and emotional_tone",
            2200,
        )
        return LocalLLMProvider._ensure_scene_text(
            self, result, story_bible, scene, source_text, preceding_context
        )

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
        return LocalLLMProvider._ensure_scene_text(
            self, result, story_bible, scene, source_text, preceding_context="", instruction=instruction
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
        return [item for item in issues if isinstance(item, dict)]
