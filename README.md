# StoryForge AI — Offline Story-to-Screenplay Studio (V8)

StoryForge is a single-process Streamlit app for turning source stories into a structured Story Bible, scene plan, automatically generated screenplay, quality findings, editable scenes, and PDF/JSON exports.

- Faster main generation flow: screenplay generation now uses a progress indicator and applies quick deterministic quality checks immediately, without blocking completion on a second full-model AI review call. The full AI reviewer remains available with **Run full quality checks + AI reviewer**.
- Scene prompts use source paragraphs linked to each scene's source event IDs when those links are available. This reduces repeated prompt context while preserving the full Story Bible and neighboring source paragraphs. When source links are missing, StoryForge falls back to the full story.
- Prior screenplay context sent to each generation call is bounded to reduce repeated prompt tokens. Generation instructions, model settings, and scene output token budget are retained.
- Quality findings expand to show the specific issue, evidence, recommended next step, and a button to select the affected scene in the editor.
- Existing SQLite databases are upgraded automatically with evidence and suggested-action columns; existing projects are retained.

## Local offline AI

In the sidebar select Ollama and a locally installed model such as `gemma3:4b`, or select a GGUF file. N0 API key is used. Local Ollama mode connects only to `http://127.0.0.1:11434`.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```
or 
```
python -m streamlit run app.py
```

## Tests

```powershell
python -m pytest -q
```
