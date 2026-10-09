# StoryForge AI V8 — Local / Offline Setup

## Run locally

1. Install Python 3.10–3.12 and the requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Start Ollama and confirm your model using `ollama list` (recommended model: `gemma3:4b`) or choose an existing GGUF file in the Streamlit sidebar.
3. Run the app:

   ```powershell
   streamlit run app.py
   ```

StoryForge uses a single Streamlit process. It does not require FastAPI, React, Node.js, an OpenAI key, or internet access during normal local inference after dependencies and model weights are present.

## Speed and quality

V8 keeps the same scene generation instructions and generation token budget. To reduce wait before the first screenplay draft is available, automatic generation runs fast deterministic checks, and the more expensive semantic AI review is a separate button under **Quality Check + Export**. Run **Run full quality checks + AI reviewer** for the deeper review. Scene generation displays per-scene progress.

For projects whose Story Bible contains `source_paragraph_ids`, generation prompts use the linked paragraphs and neighboring paragraphs rather than resending the entire story for every scene. If source traceability is unavailable, it falls back to the full story. This is intended to reduce redundant prompt tokens without changing the model or reducing the generated screenplay output budget.

## Findings

Expand each red/yellow/blue finding to read the specific issue, supporting evidence, suggested next step, and (for scene-level findings) use **Open this scene in the screenplay editor**. Existing SQLite databases are migrated automatically; do not delete `data/` or `models/` when updating.
