from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import DB_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sqlite_value(value):
    """Convert structured Python values into SQLite-safe JSON text.

    LLM output can occasionally put a list/dict in a field intended to be text.
    SQLite cannot bind lists or dictionaries directly, so serialize them instead
    of crashing the whole workflow.
    """
    if isinstance(value, (list, dict, tuple)):
        return dumps(list(value) if isinstance(value, tuple) else value)
    return value


def _execute(conn: sqlite3.Connection, sql: str, params=None):
    """Execute SQL while making bound parameters SQLite-compatible."""
    if params is None:
        return conn.execute(sql)
    return conn.execute(sql, tuple(_sqlite_value(value) for value in params))


def loads(value, default):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _execute(conn, "PRAGMA foreign_keys = ON")
    try:
        initialize(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'created',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            content_type TEXT DEFAULT '',
            storage_path TEXT NOT NULL,
            original_text TEXT NOT NULL DEFAULT '',
            normalized_text TEXT NOT NULL DEFAULT '',
            source_paragraphs TEXT NOT NULL DEFAULT '[]',
            character_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS story_bibles (
            id TEXT PRIMARY KEY,
            project_id TEXT UNIQUE NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            genres TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            narrative_tone TEXT NOT NULL DEFAULT '',
            themes TEXT NOT NULL DEFAULT '[]',
            characters TEXT NOT NULL DEFAULT '[]',
            locations TEXT NOT NULL DEFAULT '[]',
            timeline TEXT NOT NULL DEFAULT '[]',
            major_events TEXT NOT NULL DEFAULT '[]',
            open_plot_threads TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scenes (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            scene_number INTEGER NOT NULL,
            act INTEGER NOT NULL DEFAULT 1,
            sequence INTEGER NOT NULL DEFAULT 1,
            slugline TEXT NOT NULL,
            interior_or_exterior TEXT NOT NULL DEFAULT 'INT.',
            location TEXT NOT NULL DEFAULT 'UNKNOWN',
            time_of_day TEXT NOT NULL DEFAULT 'DAY',
            characters TEXT NOT NULL DEFAULT '[]',
            story_purpose TEXT NOT NULL DEFAULT '',
            source_event_ids TEXT NOT NULL DEFAULT '[]',
            action TEXT NOT NULL DEFAULT '',
            dialogue TEXT NOT NULL DEFAULT '[]',
            emotional_tone TEXT NOT NULL DEFAULT 'neutral',
            transitions TEXT NOT NULL DEFAULT '[]',
            props TEXT NOT NULL DEFAULT '[]',
            production_flags TEXT NOT NULL DEFAULT '[]',
            estimated_duration INTEGER NOT NULL DEFAULT 2,
            screenplay_text TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'planned',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_issues (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            scene_id TEXT,
            check_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            evidence TEXT NOT NULL DEFAULT '',
            suggested_action TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        """
    )
    columns = {row[1] for row in _execute(conn, "PRAGMA table_info(story_bibles)")}
    if "narrative_tone" not in columns:
        _execute(conn, "ALTER TABLE story_bibles ADD COLUMN narrative_tone TEXT NOT NULL DEFAULT ''")

    # Upgrade existing local databases without requiring users to delete their projects.
    issue_columns = {row[1] for row in _execute(conn, "PRAGMA table_info(analysis_issues)")}
    if "evidence" not in issue_columns:
        _execute(conn, "ALTER TABLE analysis_issues ADD COLUMN evidence TEXT NOT NULL DEFAULT ''")
    if "suggested_action" not in issue_columns:
        _execute(conn, "ALTER TABLE analysis_issues ADD COLUMN suggested_action TEXT NOT NULL DEFAULT ''")


def project_row(row):
    if row is None:
        return None
    return dict(row)


def document_row(row):
    if row is None:
        return None
    item = dict(row)
    item["source_paragraphs"] = loads(item.pop("source_paragraphs"), [])
    return item


def bible_row(row):
    if row is None:
        return None
    item = dict(row)
    for key in ["genres", "themes", "characters", "locations", "timeline", "major_events", "open_plot_threads"]:
        item[key] = loads(item[key], [])
    return item


def scene_row(row):
    if row is None:
        return None
    item = dict(row)
    for key in ["characters", "source_event_ids", "dialogue", "transitions", "props", "production_flags"]:
        item[key] = loads(item[key], [])
    return item


def issue_row(row):
    return dict(row) if row else None


def create_project(title: str, description: str = "") -> dict:
    project_id = str(uuid4())
    stamp = now_iso()
    with connection() as conn:
        _execute(conn, 
            "INSERT INTO projects(id,title,description,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (project_id, title, description, "created", stamp, stamp),
        )
        return project_row(_execute(conn, "SELECT * FROM projects WHERE id=?", (project_id,)).fetchone())


def list_projects() -> list[dict]:
    with connection() as conn:
        return [project_row(r) for r in _execute(conn, "SELECT * FROM projects ORDER BY updated_at DESC")]


def get_project(project_id: str) -> dict | None:
    with connection() as conn:
        return project_row(_execute(conn, "SELECT * FROM projects WHERE id=?", (project_id,)).fetchone())


def update_project_status(project_id: str, status: str) -> None:
    with connection() as conn:
        _execute(conn, "UPDATE projects SET status=?, updated_at=? WHERE id=?", (status, now_iso(), project_id))


def delete_project(project_id: str) -> None:
    with connection() as conn:
        _execute(conn, "DELETE FROM projects WHERE id=?", (project_id,))


def save_document(project_id: str, filename: str, content_type: str, storage_path: str, original_text: str, normalized_text: str, source_paragraphs: list[dict]) -> dict:
    doc_id = str(uuid4())
    with connection() as conn:
        _execute(conn, 
            """INSERT INTO documents(id,project_id,filename,content_type,storage_path,original_text,normalized_text,source_paragraphs,character_count,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (doc_id, project_id, filename, content_type or "", storage_path, original_text, normalized_text, dumps(source_paragraphs), len(normalized_text), now_iso()),
        )
        _execute(conn, "UPDATE projects SET status=?, updated_at=? WHERE id=?", ("ingested", now_iso(), project_id))
        return document_row(_execute(conn, "SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())


def latest_document(project_id: str) -> dict | None:
    with connection() as conn:
        return document_row(_execute(conn, "SELECT * FROM documents WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,)).fetchone())


def upsert_story_bible(project_id: str, bible: dict, status: str = "analyzed") -> dict:
    bible_id = str(uuid4())
    stamp = now_iso()
    with connection() as conn:
        existing = _execute(conn, "SELECT id FROM story_bibles WHERE project_id=?", (project_id,)).fetchone()
        values = (
            bible.get("title", "Untitled"),
            dumps(bible.get("genres", [])),
            bible.get("summary", ""),
            bible.get("narrative_tone", ""),
            dumps(bible.get("themes", [])),
            dumps(bible.get("characters", [])),
            dumps(bible.get("locations", [])),
            dumps(bible.get("timeline", [])),
            dumps(bible.get("major_events", [])),
            dumps(bible.get("open_plot_threads", [])),
            stamp,
        )
        if existing:
            _execute(conn, 
                """UPDATE story_bibles SET title=?,genres=?,summary=?,narrative_tone=?,themes=?,characters=?,locations=?,timeline=?,major_events=?,open_plot_threads=?,updated_at=? WHERE project_id=?""",
                values + (project_id,),
            )
        else:
            _execute(conn, 
                """INSERT INTO story_bibles(id,project_id,title,genres,summary,narrative_tone,themes,characters,locations,timeline,major_events,open_plot_threads,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (bible_id, project_id, *values),
            )
        _execute(conn, "UPDATE projects SET status=?, updated_at=? WHERE id=?", (status, stamp, project_id))
        return bible_row(_execute(conn, "SELECT * FROM story_bibles WHERE project_id=?", (project_id,)).fetchone())


def get_story_bible(project_id: str) -> dict | None:
    with connection() as conn:
        return bible_row(_execute(conn, "SELECT * FROM story_bibles WHERE project_id=?", (project_id,)).fetchone())


def replace_scenes(project_id: str, scenes: list[dict]) -> list[dict]:
    stamp = now_iso()
    with connection() as conn:
        _execute(conn, "DELETE FROM scenes WHERE project_id=?", (project_id,))
        for scene in scenes:
            scene_id = scene.get("id", str(uuid4()))
            _execute(conn, 
                """INSERT INTO scenes(id,project_id,scene_number,act,sequence,slugline,interior_or_exterior,location,time_of_day,characters,story_purpose,source_event_ids,action,dialogue,emotional_tone,transitions,props,production_flags,estimated_duration,screenplay_text,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    scene_id, project_id, scene["scene_number"], scene["act"], scene["sequence"], scene["slugline"], scene["interior_or_exterior"],
                    scene["location"], scene["time_of_day"], dumps(scene.get("characters", [])), scene.get("story_purpose", ""), dumps(scene.get("source_event_ids", [])),
                    scene.get("action", ""), dumps(scene.get("dialogue", [])), scene.get("emotional_tone", "neutral"), dumps(scene.get("transitions", [])), dumps(scene.get("props", [])),
                    dumps(scene.get("production_flags", [])), scene.get("estimated_duration", 2), scene.get("screenplay_text", ""), scene.get("status", "planned"), stamp, stamp
                ),
            )
        _execute(conn, "UPDATE projects SET status=?, updated_at=? WHERE id=?", ("scene_plan_ready", stamp, project_id))
        return list_scenes_in_conn(conn, project_id)


def list_scenes_in_conn(conn, project_id):
    return [scene_row(r) for r in _execute(conn, "SELECT * FROM scenes WHERE project_id=? ORDER BY scene_number", (project_id,))]


def list_scenes(project_id: str) -> list[dict]:
    with connection() as conn:
        return list_scenes_in_conn(conn, project_id)


def update_scene(scene_id: str, values: dict, status: str | None = None) -> dict:
    allowed = {
        "scene_number", "act", "sequence", "slugline", "interior_or_exterior", "location", "time_of_day", "characters",
        "story_purpose", "source_event_ids", "action", "dialogue", "emotional_tone", "transitions", "props", "production_flags",
        "estimated_duration", "screenplay_text", "status"
    }
    assignments = []
    params = []
    json_keys = {"characters", "source_event_ids", "dialogue", "transitions", "props", "production_flags"}
    for key, value in values.items():
        if key not in allowed:
            continue
        assignments.append(f"{key}=?")
        params.append(dumps(value) if key in json_keys else value)
    if status:
        assignments.append("status=?")
        params.append(status)
    assignments.append("updated_at=?")
    params.append(now_iso())
    params.append(scene_id)
    with connection() as conn:
        _execute(conn, f"UPDATE scenes SET {', '.join(assignments)} WHERE id=?", params)
        return scene_row(_execute(conn, "SELECT * FROM scenes WHERE id=?", (scene_id,)).fetchone())


def replace_quality_issues(project_id: str, issues: list[dict]) -> list[dict]:
    stamp = now_iso()
    with connection() as conn:
        _execute(conn, "DELETE FROM analysis_issues WHERE project_id=?", (project_id,))
        for issue in issues:
            _execute(conn, 
                "INSERT INTO analysis_issues(id,project_id,scene_id,check_type,severity,message,evidence,suggested_action,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    issue.get("id", str(uuid4())), project_id, issue.get("scene_id"),
                    issue["check_type"], issue["severity"], issue["message"],
                    issue.get("evidence", ""), issue.get("suggested_action", ""), stamp,
                ),
            )
        _execute(conn, "UPDATE projects SET status=?, updated_at=? WHERE id=?", ("quality_checked", stamp, project_id))
        return [issue_row(r) for r in _execute(conn, "SELECT * FROM analysis_issues WHERE project_id=? ORDER BY created_at", (project_id,))]


def list_quality_issues(project_id: str) -> list[dict]:
    with connection() as conn:
        return [issue_row(r) for r in _execute(conn, "SELECT * FROM analysis_issues WHERE project_id=? ORDER BY created_at", (project_id,))]
