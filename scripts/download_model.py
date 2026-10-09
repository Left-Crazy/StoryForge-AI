from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
TARGET = MODELS / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"


def main() -> int:
    if TARGET.exists() and TARGET.stat().st_size > 300_000_000:
        print(f"Model already present: {TARGET}")
        return 0
    print("Downloading the offline StoryForge model (~400 MB)...")
    print("This is the only step that needs internet access. After it finishes, StoryForge can run offline.")
    request = Request(URL, headers={"User-Agent": "StoryForge-AI/1.0"})
    try:
        with urlopen(request, timeout=60) as response, TARGET.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\rProgress: {pct:5.1f}%", end="", flush=True)
        print("\nModel saved to:", TARGET)
        return 0
    except Exception as exc:
        if TARGET.exists():
            TARGET.unlink(missing_ok=True)
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
