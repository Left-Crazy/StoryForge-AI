from __future__ import annotations

import json

def text_items(value, kind: str = "generic") -> list[str]:
    """Convert model-produced list values into safe, human-readable strings.

    Older/weak local-model responses sometimes encode simple lists (especially
    locations, genres, themes, and plot threads) as objects instead of strings.
    This helper also makes already-saved projects render safely.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.splitlines() if "\n" in value else [value]
    elif isinstance(value, dict):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    preferred = {
        "location": ("name", "location", "place", "title", "label", "description", "summary", "text"),
        "genre": ("name", "genre", "label", "title", "description", "text"),
        "theme": ("name", "theme", "label", "title", "description", "text"),
        "thread": ("thread", "question", "title", "description", "summary", "text", "name"),
        "generic": ("name", "title", "label", "description", "summary", "text", "value"),
    }.get(kind, ("name", "title", "label", "description", "summary", "text", "value"))

    result: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        candidate = item
        if isinstance(item, dict):
            candidate = next((item.get(key) for key in preferred if item.get(key) not in (None, "")), None)
            if candidate is None:
                candidate = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if isinstance(candidate, (dict, list, tuple)):
            candidate = json.dumps(candidate, ensure_ascii=False)
        text = str(candidate).strip()
        if text and text not in result:
            result.append(text)
    return result


def joined_text(value, separator: str = ", ", kind: str = "generic") -> str:
    """Join an arbitrary model-produced list without raising on dict items."""
    return separator.join(text_items(value, kind=kind))

