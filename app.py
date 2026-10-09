from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from storyforge import db
from storyforge.ai.ollama_provider import OllamaProvider
from storyforge.services import documents, export, projects, quality, scenes, story
from storyforge.ui.helpers import joined_text, jtext, parse_json, safe_error, status_label, text_items

st.set_page_config(
    page_title="StoryForge AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1450px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .hero { padding: 1.15rem 1.35rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1rem; }
    .muted { opacity: .72; }
    .step { border-left: 4px solid #7c83ff; padding-left: .75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "session_model_path" not in st.session_state:
    st.session_state.session_model_path = os.getenv(
        "STORYFORGE_MODEL_PATH",
        "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
if "ollama_model_name" not in st.session_state:
    st.session_state.ollama_model_name = os.getenv("STORYFORGE_OLLAMA_MODEL", "gemma3:4b")
if "ai_backend_choice" not in st.session_state:
    st.session_state.ai_backend_choice = (
        "GGUF file (llama.cpp)" if os.getenv("STORYFORGE_AI_BACKEND", "ollama").lower() == "gguf"
        else "Ollama (recommended if installed)"
    )


def model_config() -> str:
    return st.session_state.get("active_model_config", st.session_state.session_model_path).strip()

def current_project():
    return db.get_project(st.session_state.project_id) if st.session_state.project_id else None


def refresh():
    st.rerun()


def focus_editor(project_id: str, scene_id: str) -> None:
    """Callback: switch the editor selection before Streamlit reruns the app."""
    st.session_state[f"editor_{project_id}"] = scene_id


with st.sidebar:
    st.header("StoryForge AI")
    st.caption("Single-process Streamlit architecture")

    st.subheader("Offline AI")
    st.caption("Both options run locally. No OpenAI API key is used. Ollama is recommended when you already have Gemma/Qwen installed.")
    backend_choice = st.radio(
        "Local inference engine",
        ["Ollama (recommended if installed)", "GGUF file (llama.cpp)"],
        key="ai_backend_choice",
        help="Ollama reuses models installed with `ollama pull`. GGUF loads a .gguf file directly.",
    )
    if backend_choice == "Ollama (recommended if installed)":
        ollama_model = st.text_input(
            "Installed Ollama model name",
            key="ollama_model_name",
            help="For your computer, try gemma3:4b. The smaller qwen2.5:0.5b may struggle with JSON-heavy tasks.",
        ).strip()
        model_path = f"ollama:{ollama_model}" if ollama_model else "ollama:"
        st.session_state.active_model_config = model_path
        if ollama_model:
            with st.spinner("Checking local Ollama…"):
                ready, status = OllamaProvider(ollama_model).check_model()
            if ready:
                st.success(status)
            else:
                st.warning(status)
                st.caption("If Ollama is installed, open the Ollama desktop app. Check model names with `ollama list`.")
        provider_display = f"Ollama Local AI · {ollama_model or 'no model selected'}"
    else:
        model_path = st.text_input(
            "Local GGUF model path",
            value=st.session_state.session_model_path,
            key="session_model_path",
            help="Use the full path or a path relative to this project folder.",
        ).strip()
        st.session_state.active_model_config = model_path
        model_check_path = Path(model_path)
        if not model_check_path.is_absolute():
            model_check_path = Path(__file__).resolve().parent / model_check_path
        if model_check_path.exists():
            st.success("Local GGUF model file found")
        else:
            st.warning("Local GGUF model not found. Enter the exact .gguf file path.")
        provider_display = f"Local GGUF AI · {model_path}"

    st.divider()
    st.header("Projects")
    all_projects = projects.list_all()
    options = ["— New project —"] + [f"{p['title']}  ·  {status_label(p['status'])}" for p in all_projects]
    current_index = 0
    if st.session_state.project_id:
        for i, p in enumerate(all_projects, 1):
            if p["id"] == st.session_state.project_id:
                current_index = i
                break
    selected = st.selectbox("Open project", options, index=current_index)
    if selected == "— New project —":
        # Use ordinary widgets + a normal button rather than a form. This avoids
        # Streamlit's missing-submit warning on older/local Streamlit installs.
        title = st.text_input("Project title", placeholder="My Short Film", key="new_project_title")
        description = st.text_area("Description", placeholder="Optional project description", key="new_project_description")
        if st.button("Create project", type="primary", use_container_width=True, key="create_project_button"):
            try:
                if not title.strip():
                    raise ValueError("Enter a project title.")
                created = projects.create(title.strip(), description.strip())
                st.session_state.project_id = created["id"]
                refresh()
            except Exception as exc:
                safe_error(exc)
    else:
        chosen = next(p for p in all_projects if f"{p['title']}  ·  {status_label(p['status'])}" == selected)
        st.session_state.project_id = chosen["id"]
        st.caption(f"Status: **{status_label(chosen['status'])}**")
        if st.button("Delete current project", use_container_width=True):
            projects.delete(chosen["id"])
            st.session_state.project_id = None
            refresh()

project = current_project()

st.markdown(
    '<div class="hero"><h1>🎬 StoryForge AI</h1>'
    '<p>Turn a story into a structured, editable, quality-checked screenplay — all inside one Streamlit app.</p>'
    '<p class="muted">Story → Story Bible → Scene Plan → Automatic Screenplay → Quality Check → Editor → PDF / JSON</p></div>',
    unsafe_allow_html=True,
)

if not project:
    st.info("Create a project from the sidebar to begin.")
    cols = st.columns(4)
    cards = [
        ("1. Ingestion", "PDF, DOCX, TXT, or pasted text"),
        ("2. Story Bible", "Characters, locations, events, timeline, themes"),
        ("3. Screenplay", "AI scene planning and automatic scene generation"),
        ("4. Quality + Export", "Continuity checks, editor, PDF and JSON"),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.subheader(title)
            st.caption(body)
    st.stop()

project_id = project["id"]
model_path = model_config()

# -----------------------------------------------------------------------------
# 1. INGESTION
# -----------------------------------------------------------------------------
st.header("1. Story Ingestion")
st.markdown('<div class="step">Give StoryForge the source story. The source remains available for traceability during AI generation.</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    upload = st.file_uploader("Upload story", type=["txt", "pdf", "docx"], key=f"upload_{project_id}")
    if upload and st.button("Add uploaded story", type="primary", key=f"upload_btn_{project_id}"):
        try:
            doc = documents.ingest_upload(project_id, upload.name, upload.type, upload.getvalue())
            st.success(f"Uploaded {doc['filename']} — {doc['character_count']:,} normalized characters.")
            refresh()
        except Exception as exc:
            safe_error(exc)
with right:
    pasted_title = st.text_input("Pasted story title", value="pasted-story.txt", key=f"paste_title_{project_id}")
    pasted = st.text_area("Or paste story text", height=180, key=f"paste_text_{project_id}")
    if st.button("Save pasted story", key=f"paste_btn_{project_id}"):
        if not pasted.strip():
            st.warning("Paste some story text first.")
        else:
            try:
                documents.ingest_text(project_id, pasted_title.strip() or "pasted-story.txt", pasted)
                st.success("Story saved.")
                refresh()
            except Exception as exc:
                safe_error(exc)

doc = documents.latest(project_id)
if doc:
    with st.expander("Current source and traceability", expanded=False):
        st.write(f"**Source:** {doc['filename']} · {doc['character_count']:,} normalized characters · {len(doc['source_paragraphs'])} paragraphs")
        st.code(doc["normalized_text"][:5000], language="text")

# -----------------------------------------------------------------------------
# 2. STORY BIBLE
# -----------------------------------------------------------------------------
st.header("2. Story Bible")
if doc and st.button("Analyze story with AI", type="primary", key=f"analyze_{project_id}"):
    with st.spinner("Understanding characters, events, locations, themes and timeline…"):
        try:
            result = story.analyze(project_id, model_path)
            st.success("Story Bible generated by the local offline AI model.")
            st.session_state.project_id = project_id
            st.write(f"Detected **{len(result.get('characters', []))} characters**, **{len(result.get('major_events', []))} major events**, and **{len(result.get('locations', []))} locations**.")
            refresh()
        except Exception as exc:
            safe_error(exc)

bible = story.get(project_id)
if bible:
    c1, c2 = st.columns(2)
    with c1:
        btitle = st.text_input("Title", bible.get("title", ""), key=f"bible_title_{project_id}")
        genres = st.text_input(
            "Genres (comma-separated)", joined_text(bible.get("genres", []), kind="genre"),
            key=f"bible_genres_{project_id}",
        )
        themes = st.text_input(
            "Themes (comma-separated)", joined_text(bible.get("themes", []), kind="theme"),
            key=f"bible_themes_{project_id}",
        )
        tone = st.text_input("Narrative tone", bible.get("narrative_tone", ""), key=f"bible_tone_{project_id}")
        summary = st.text_area("Summary", bible.get("summary", ""), height=170, key=f"bible_summary_{project_id}")
        locations = st.text_area(
            "Locations (one per line)",
            "\n".join(text_items(bible.get("locations", []), kind="location")),
            height=100, key=f"bible_locations_{project_id}",
        )
        threads = st.text_area(
            "Open plot threads (one per line)",
            "\n".join(text_items(bible.get("open_plot_threads", []), kind="thread")),
            height=100, key=f"bible_threads_{project_id}",
        )
    with c2:
        characters = st.text_area("Characters (JSON)", jtext(bible.get("characters", [])), height=250, key=f"bible_characters_{project_id}")
        timeline = st.text_area("Timeline (JSON)", jtext(bible.get("timeline", [])), height=170, key=f"bible_timeline_{project_id}")
        major_events = st.text_area("Major events (JSON)", jtext(bible.get("major_events", [])), height=220, key=f"bible_events_{project_id}")

    # No st.form here: ordinary widgets are saved explicitly with this button.
    if st.button("Save Story Bible", type="primary", key=f"save_bible_{project_id}"):
        try:
            payload = {
                "title": btitle.strip() or project["title"],
                "genres": [x.strip() for x in genres.split(",") if x.strip()],
                "summary": summary.strip(),
                "themes": [x.strip() for x in themes.split(",") if x.strip()],
                "narrative_tone": tone.strip(),
                "characters": parse_json(characters, "Characters"),
                "locations": [x.strip() for x in locations.splitlines() if x.strip()],
                "timeline": parse_json(timeline, "Timeline"),
                "major_events": parse_json(major_events, "Major events"),
                "open_plot_threads": [x.strip() for x in threads.splitlines() if x.strip()],
            }
            story.update(project_id, payload)
            st.success("Story Bible saved.")
            refresh()
        except Exception as exc:
            safe_error(exc)
else:
    st.info("Upload or paste a story, then analyze it to create the Story Bible.")

# -----------------------------------------------------------------------------
# 3. SCENE PLAN + AUTOMATIC SCREENPLAY GENERATION
# -----------------------------------------------------------------------------
st.header("3. Scene Plan → Automatic Screenplay")
if bible:
    st.caption("The primary workflow does not ask the user to write the first screenplay draft. StoryForge plans the scenes and then generates every scene automatically.")
    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Build Scene Plan + Generate Screenplay", type="primary", use_container_width=True, key=f"build_generate_{project_id}"):
            try:
                progress_bar = st.progress(0, text="Preparing scene plan…")
                progress_message = st.empty()

                def update_generation_progress(stage: str, completed: int, total: int, message: str) -> None:
                    progress_message.info(message)
                    if stage == "planning":
                        progress_bar.progress(0, text="Planning scenes…")
                    elif stage == "planned":
                        progress_bar.progress(0, text=f"Scene plan ready · {total} scenes")
                    elif total:
                        fraction = min(1.0, max(0.0, completed / total))
                        progress_bar.progress(fraction, text=f"Generating screenplay · {completed}/{total} scenes")

                generated = scenes.build_plan_and_generate(
                    project_id, model_path, progress_callback=update_generation_progress
                )
                # Don't add a second slow LLM request to the critical generation path.
                # Fast deterministic checks run immediately; the complete AI semantic
                # review is available below as a separate explicit action.
                issues = quality.run(project_id, model_path, include_ai_review=False)
                progress_bar.progress(1.0, text="Screenplay generation complete")
                progress_message.success("Fast deterministic checks complete. Run the deeper AI review below when ready.")
                st.success(f"Generated {len(generated)} screenplay scenes and ran the fast checks ({len(issues)} findings).")
                st.info("To reduce waiting, the separate AI semantic review is no longer run automatically after generation. It remains available under Quality Check + Export.")
                refresh()
            except Exception as exc:
                safe_error(exc)
    with c2:
        if st.button("Rebuild Scene Plan Only", use_container_width=True, key=f"planonly_{project_id}"):
            try:
                planned = scenes.plan(project_id, model_path)
                st.success(f"Created {len(planned)} planned scenes. Screenplay generation has not run yet.")
                refresh()
            except Exception as exc:
                safe_error(exc)

scene_list = scenes.list_all(project_id)
if scene_list:
    st.subheader(f"Scene Plan · {len(scene_list)} scenes")
    for scene in scene_list:
        with st.expander(f"Scene {scene['scene_number']} — {scene['slugline']}", expanded=scene["scene_number"] == 1):
            cols = st.columns(4)
            cols[0].metric("Act", scene["act"])
            cols[1].metric("Sequence", scene["sequence"])
            cols[2].metric("Duration", f"{scene['estimated_duration']} min")
            cols[3].metric("Status", status_label(scene["status"]))
            st.write(f"**Purpose:** {scene['story_purpose']}")
            st.write(f"**Characters:** {joined_text(scene.get('characters', [])) or 'None'}")
            st.write(f"**Source events:** {joined_text(scene.get('source_event_ids', [])) or 'None'}")
            if not scene.get("screenplay_text"):
                if st.button("Generate this missing scene", key=f"generate_{scene['id']}"):
                    try:
                        with st.spinner(f"Generating Scene {scene['scene_number']}…"):
                            scenes.generate_one(project_id, scene["id"], model_path)
                        refresh()
                    except Exception as exc:
                        safe_error(exc)
else:
    st.info("Create the Story Bible first, then build the scene plan.")

# -----------------------------------------------------------------------------
# 4. SCREENPLAY EDITOR
# -----------------------------------------------------------------------------
st.header("4. Screenplay Editor")
scene_list = scenes.list_all(project_id)
if scene_list:
    labels = {s["id"]: f"Scene {s['scene_number']} — {s['slugline']}" for s in scene_list}
    selected_id = st.selectbox("Scene", list(labels), format_func=lambda x: labels[x], key=f"editor_{project_id}")
    selected_scene = next(s for s in scene_list if s["id"] == selected_id)

    if not selected_scene.get("screenplay_text"):
        st.warning("This scene has not been generated yet. Generate it from the Scene Plan above.")
    else:
        left, right = st.columns([2.2, 1])
        with left:
            st.subheader("Generated screenplay")
            screenplay_text = st.text_area("Edit screenplay", selected_scene.get("screenplay_text", ""), height=600, key=f"screenplay_{selected_id}")
            action = st.text_area("Scene action summary", selected_scene.get("action", ""), height=130, key=f"action_{selected_id}")
            if st.button("Save manual edits", type="primary", key=f"save_{selected_id}"):
                try:
                    scenes.save_edits(selected_id, action, screenplay_text)
                    st.success("Scene saved.")
                    refresh()
                except Exception as exc:
                    safe_error(exc)

        with right:
            st.subheader("Targeted AI revision")
            instruction = st.text_area(
                "Instruction",
                placeholder="Increase tension, make dialogue more natural, shorten, expand, make darker…",
                height=150,
                key=f"instruction_{selected_id}",
            )
            if st.button("Revise scene with AI", type="secondary", use_container_width=True, key=f"revise_{selected_id}"):
                try:
                    with st.spinner("Revising the complete scene while preserving story facts…"):
                        scenes.revise(project_id, selected_id, instruction, model_path)
                    st.success("Scene revised.")
                    refresh()
                except Exception as exc:
                    safe_error(exc)

            st.subheader("Scene metadata")
            st.write(f"**Tone:** {selected_scene.get('emotional_tone', '—')}")
            st.write(f"**Props:** {joined_text(selected_scene.get('props', [])) or 'None'}")
            st.write(f"**Production flags:** {joined_text(selected_scene.get('production_flags', [])) or 'None'}")
            st.write(f"**Source events:** {', '.join(selected_scene.get('source_event_ids', [])) or 'None'}")
else:
    st.info("Once the screenplay is generated, the editor will open with the generated text already populated.")

# -----------------------------------------------------------------------------
# 5. QUALITY + EXPORT
# -----------------------------------------------------------------------------
st.header("5. Quality Check + Export")
qc1, qc2, qc3 = st.columns(3)
with qc1:
    if st.button("Run full quality checks + AI reviewer", type="primary", use_container_width=True, key=f"quality_{project_id}"):
        try:
            with st.spinner("Reviewing continuity, source fidelity, pacing and production details with local AI…"):
                issues = quality.run(project_id, model_path)
            st.success(f"Quality check complete: {len(issues)} findings.")
            refresh()
        except Exception as exc:
            safe_error(exc)
with qc2:
    if st.button("Export PDF", use_container_width=True, key=f"pdf_{project_id}"):
        try:
            pdf_bytes = export.project_pdf(project_id)
            st.download_button("Download screenplay PDF", pdf_bytes, file_name=f"{project['title']}.pdf", mime="application/pdf", key=f"download_pdf_{project_id}")
        except Exception as exc:
            safe_error(exc)
with qc3:
    if st.button("Prepare JSON export", use_container_width=True, key=f"json_{project_id}"):
        try:
            json_bytes = export.project_json(project_id)
            st.download_button("Download project JSON", json_bytes, file_name=f"{project['title']}.json", mime="application/json", key=f"download_json_{project_id}")
        except Exception as exc:
            safe_error(exc)

issues = db.list_quality_issues(project_id)
if issues:
    st.subheader(f"Quality findings · {len(issues)}")
    scene_by_id = {scene["id"]: scene for scene in scenes.list_all(project_id)}
    st.caption("Each finding can be expanded to see why it was flagged, supporting evidence, a suggested next step, and a shortcut to its scene.")
    for issue in issues:
        severity = issue["severity"].lower()
        scene = scene_by_id.get(issue.get("scene_id"))
        scene_label = (
            f"Scene {scene['scene_number']} · {scene['slugline']}" if scene
            else "Project-wide finding"
        )
        badge = {"error": "🔴 ERROR", "warning": "🟡 WARNING", "info": "🔵 INFO"}.get(severity, "ℹ️ INFO")
        message = issue.get("message") or "The reviewer did not return a message. Review the evidence and suggested action below."
        if severity == "error":
            expander_label = f"{badge} · {scene_label} · {issue['check_type'].replace('_', ' ').title()} · {message[:100]}"
        elif severity == "warning":
            expander_label = f"{badge} · {scene_label} · {issue['check_type'].replace('_', ' ').title()} · {message[:100]}"
        else:
            expander_label = f"{badge} · {scene_label} · {issue['check_type'].replace('_', ' ').title()} · {message[:100]}"
        with st.expander(expander_label, expanded=False):
            if severity == "error":
                st.error(message)
            elif severity == "warning":
                st.warning(message)
            else:
                st.info(message)
            evidence = str(issue.get("evidence") or "").strip()
            suggested_action = str(issue.get("suggested_action") or quality.default_action(issue.get("check_type", ""), message)).strip()
            if evidence:
                st.markdown("**Why it was flagged / evidence**")
                st.write(evidence)
            if suggested_action:
                st.markdown("**Suggested next step**")
                st.write(suggested_action)
            if ("possible issue" in message.lower() or "did not return a message" in message.lower()) and not evidence:
                st.info("This finding may have been saved by an earlier version without evidence or a concrete next step. Run the full AI reviewer again to refresh its explanation.")
            if scene:
                st.caption(f"Affected scene: {scene_label}")
                st.button(
                    "Open this scene in the screenplay editor",
                    key=f"open_issue_scene_{project_id}_{issue['id']}",
                    on_click=focus_editor,
                    args=(project_id, scene["id"]),
                    help="Selects the affected scene in the Screenplay Editor above.",
                )
else:
    st.caption("No quality findings saved yet. Run the quality checks after generation or revision.")

st.divider()
st.caption(
    f"AI provider: {provider_display} · offline/local inference · "
    "StoryForge services run inside one Streamlit process."
)
