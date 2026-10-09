from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Provider boundary for StoryForge's AI operations."""

    name = "base"
    is_real_ai = False

    @abstractmethod
    def analyze_story(self, text: str, title: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def plan_scenes(self, story_bible: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def generate_scene(
        self,
        story_bible: dict[str, Any],
        scene: dict[str, Any],
        source_text: str,
        preceding_context: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def revise_scene(
        self,
        story_bible: dict[str, Any],
        scene: dict[str, Any],
        instruction: str,
        source_text: str,
    ) -> dict[str, Any]:
        raise NotImplementedError
