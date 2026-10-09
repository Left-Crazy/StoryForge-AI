from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("STORYFORGE_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "storyforge.db"
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
