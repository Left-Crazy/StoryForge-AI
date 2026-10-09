from __future__ import annotations

import json


def _clip(text: str, limit: int = 10000) -> str:
    """Keep the opening and ending beats while bounding CPU inference time."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    tail = limit - head
    return text[:head] + "\n\n[Middle section omitted to fit local model context.]\n\n" + text[-tail:]


def story_analysis_prompt(source_text: str, title: str) -> str:
    return f"""
You are the story understanding engine for StoryForge AI, a story-to-screenplay application.

Analyze the source story below and create a faithful Story Bible.

Rules:
- Ground every fact in the supplied story. Do not invent named characters, locations, events,
  relationships, or backstory that is not supported.
- Short stories are valid inputs. Extract what is present instead of declaring the story too short.
- Preserve the order of events.
- Use source paragraph IDs such as p-1, p-2 in source references.
- If a fact is uncertain or ambiguous, say so in the relevant field rather than guessing.
- Identify 1-3 likely genres and the story's narrative tone.
- Build a useful character arc even for a short story, but stay grounded in the source.
- Major events should be screenplay-relevant beats, not just a copy of every sentence.

Project title: {title}

SOURCE STORY:
{_clip(source_text)}
""".strip()


def scene_plan_prompt(story_bible: dict, source_text: str) -> str:
    bible_json = json.dumps(story_bible, ensure_ascii=False, indent=2)
    return f"""
You are a professional screenplay scene planner.

Turn the Story Bible and source story into a reviewable scene plan before prose generation.

Rules:
- Every planned scene must advance the story.
- Preserve the source's causal order and ending.
- Use 3-6 scenes for a very short story, 5-10 for a medium story, and more only when the source
  truly needs it. Do not collapse a whole short story into a single scene unless it is genuinely
  one continuous dramatic moment.
- Split beats into distinct scenes when there is a meaningful change of location, time, objective,
  relationship pressure, discovery, or escalation.
- Reference only real Story Bible characters and locations.
- source_event_ids must exactly match event IDs from the Story Bible when possible.
- Choose plausible INT./EXT. and time-of-day labels only when supported by the story; otherwise use
  UNSPECIFIED.
- Estimate each scene's screenplay duration in minutes.
- Add useful production flags only when supported or strongly implied (night, VFX, crowd, vehicle,
  stunt, prop, music, etc.).

STORY BIBLE:
{bible_json}

SOURCE STORY:
{_clip(source_text)}
""".strip()


def scene_generation_prompt(story_bible: dict, scene: dict, source_text: str, preceding_context: str) -> str:
    bible_json = json.dumps(story_bible, ensure_ascii=False, indent=2)
    scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
    return f"""
You are the screenplay generation engine for StoryForge AI.

Generate ONE complete screenplay scene from the supplied scene plan, Story Bible, source story, and
preceding scene context.

Rules:
- The source story is authoritative. Do not invent major plot facts, new named characters, or a new
  ending.
- Expand brief narrative prose into visually filmable action and natural dialogue. Do not merely repeat
  the source sentence-by-sentence.
- A short story still needs a real scene: establish the space, show behavior, dramatize the conflict,
  and land the scene on a meaningful beat.
- Preserve character names and known relationships exactly.
- Use professional screenplay conventions: slugline, concise action paragraphs, character names in
  uppercase, optional parentheticals, dialogue, and transitions only when appropriate.
- Keep action visual and present tense.
- Dialogue should reveal information through conflict/subtext instead of exposition dumping.
- Make the scene proportionate to its estimated duration. A 2-minute scene may still contain multiple
  action beats and several dialogue exchanges.
- REQUIRED: screenplay_text must be a non-empty string containing the complete formatted scene.
- Return every requested key, even if some values are empty. Never omit screenplay_text.
- screenplay_text must include the slugline, action, dialogue where appropriate, and a meaningful
  scene ending. It must agree with action/dialogue fields.
- Do not include analysis, explanations, or markdown fences in screenplay_text.

Return exactly one JSON object with these keys:
{{"action":"visual present-tense action", "dialogue":[{{"character":"CHARACTER NAME", "text":"Dialogue line"}}],
  "screenplay_text":"INT. LOCATION - DAY\\n\\nFormatted action and dialogue...", "transitions":[],
  "props":[], "production_flags":[], "emotional_tone":"..."}}

STORY BIBLE:
{bible_json}

CURRENT SCENE PLAN:
{scene_json}

SOURCE STORY:
{_clip(source_text)}

PRECEDING SCREENPLAY CONTEXT:
{preceding_context[-5000:] if preceding_context else "(This is the opening scene.)"}
""".strip()


def scene_text_only_prompt(
    story_bible: dict,
    scene: dict,
    source_text: str,
    preceding_context: str = "",
    instruction: str = "",
) -> str:
    """Compact recovery prompt used when a model omits the screenplay_text field."""
    characters = story_bible.get("characters", [])
    character_names = []
    for item in characters if isinstance(characters, list) else []:
        if isinstance(item, dict) and item.get("name"):
            character_names.append(str(item["name"]))
        elif isinstance(item, str) and item.strip():
            character_names.append(item.strip())

    compact_scene = {
        key: scene.get(key)
        for key in (
            "scene_number", "slugline", "location", "time_of_day", "characters",
            "story_purpose", "emotional_tone", "estimated_duration", "action", "dialogue",
        )
        if scene.get(key) is not None
    }
    revision = f"\nREVISION INSTRUCTION:\n{instruction.strip()}\n" if instruction.strip() else ""
    return f"""
Write the complete screenplay text for this ONE scene. This is a recovery request because a previous
response omitted the screenplay_text field.

Output one JSON object only, using exactly this shape:
{{"screenplay_text":"INT. LOCATION - DAY\\n\\nVisual action in present tense.\\n\\nCHARACTER\\nDialogue."}}

The screenplay_text value is REQUIRED and must not be empty. Include the slugline, visual action, and
natural dialogue if the scene calls for it. Use real characters from the list; do not invent major
plot facts. Do not return an outline, summary, scene purpose, or explanation. Return the formatted
screenplay, not just a description of what the scene should do.

Project: {story_bible.get('title', 'Untitled Story')}
Known characters: {json.dumps(character_names[:12], ensure_ascii=False)}
Scene plan: {json.dumps(compact_scene, ensure_ascii=False)}
{revision}
Relevant source story:
{_clip(source_text, 6000)}

Preceding screenplay context (avoid repeating it):
{_clip(preceding_context, 1500) if preceding_context else '(Opening scene.)'}
""".strip()


def revision_prompt(story_bible: dict, scene: dict, instruction: str, source_text: str) -> str:
    bible_json = json.dumps(story_bible, ensure_ascii=False, indent=2)
    scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
    return f"""
Revise the existing screenplay scene according to the user's instruction.

Rules:
- Preserve all established story facts, character identities, location, timeline, and causality unless
  the instruction explicitly asks for a stylistic change.
- Never remove a critical source fact simply to make the scene shorter.
- Apply the requested change throughout the scene, not only in one sentence.
- Return a complete replacement scene, not commentary about what changed.
- Keep screenplay_text synchronized with action and dialogue fields.

USER INSTRUCTION:
{instruction}

STORY BIBLE:
{bible_json}

CURRENT SCENE:
{scene_json}

SOURCE STORY:
{_clip(source_text)}
""".strip()


def quality_prompt(story_bible: dict, scenes: list[dict], source_text: str) -> str:
    # The semantic reviewer needs enough text to evaluate continuity, but sending
    # every verbose scene metadata field and unbounded screenplay duplicates can
    # make local models spend longer reading than reviewing.
    compact_scenes = []
    for scene in scenes:
        text = str(scene.get("screenplay_text", "") or "")
        if len(text) > 2400:
            text = _clip(text, 2400)
        compact_scenes.append({
            "scene_number": scene.get("scene_number"),
            "slugline": scene.get("slugline"),
            "location": scene.get("location"),
            "characters": scene.get("characters", []),
            "story_purpose": scene.get("story_purpose", ""),
            "source_event_ids": scene.get("source_event_ids", []),
            "action": scene.get("action", ""),
            "screenplay_text": text,
        })
    return f"""
You are the continuity and quality reviewer for StoryForge AI.

Review the generated screenplay against the Story Bible and source story.

Check for:
- character consistency
- location consistency
- timeline contradictions
- unresolved or accidentally resolved plot threads
- unsupported major facts
- missing or duplicated story beats
- dialogue/action balance and obvious pacing problems
- production flags that should be surfaced

Only report meaningful, specific issues. Do not invent problems. Use severity values: error, warning, info.
Use scene_number 0 for a project-level issue. Every issue MUST explain the actual problem in the message;
never return generic placeholders such as "AI reviewer flagged a possible issue".
For each issue, include concise source evidence and a concrete suggested_action the writer can take.
Return only this JSON shape:
{{"issues":[{{"severity":"warning", "scene_number":1, "check_type":"character_consistency",
"message":"A specific problem in this scene", "evidence":"The relevant conflicting source/scene fact",
"suggested_action":"A concrete change to resolve it"}}]}}
If no meaningful issue exists, return {{"issues":[]}}.

STORY BIBLE:
{json.dumps(story_bible, ensure_ascii=False, indent=2)}

SCENES:
{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}

SOURCE STORY:
{_clip(source_text)}
""".strip()
