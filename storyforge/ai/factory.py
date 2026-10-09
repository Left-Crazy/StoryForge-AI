from __future__ import annotations

import os
from functools import lru_cache

from .heuristic_provider import HeuristicProvider
from .local_provider import LocalLLMProvider
from .ollama_provider import OllamaProvider


def _provider_key(configured: str) -> str:
    return configured.strip()


@lru_cache(maxsize=4)
def _cached_provider(configured: str):
    """Cache model instances: Streamlit reruns the script after every interaction."""
    if configured.lower().startswith("ollama:"):
        model_name = configured.split(":", 1)[1].strip()
        return OllamaProvider(model_name=model_name)
    return LocalLLMProvider(configured or None)


def clear_provider_cache() -> None:
    """Useful after changing a model file/configuration during development."""
    _cached_provider.cache_clear()


def get_provider(model_path: str | None = None, allow_demo_fallback: bool = False):
    configured = _provider_key(model_path or os.getenv("STORYFORGE_MODEL_PATH", ""))
    if not configured:
        configured = os.getenv("STORYFORGE_MODEL_PATH", "")
    try:
        return _cached_provider(configured)
    except Exception:
        if allow_demo_fallback or os.getenv("STORYFORGE_ALLOW_DEMO_FALLBACK", "0") == "1":
            return HeuristicProvider()
        raise
