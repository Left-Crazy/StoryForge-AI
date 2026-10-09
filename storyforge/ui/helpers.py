from __future__ import annotations

import json
import streamlit as st

from ..text_utils import joined_text, text_items



def jtext(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_json(value: str, label: str):
    try:
        return json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON: {exc}")


def status_label(status: str) -> str:
    return status.replace("_", " ").title()


def safe_error(error: Exception):
    st.error(str(error))
