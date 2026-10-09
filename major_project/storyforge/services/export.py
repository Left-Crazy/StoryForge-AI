from __future__ import annotations

import json
import textwrap
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .. import db
from ..text_utils import joined_text
from .story import get as get_story_bible


def project_json(project_id: str) -> bytes:
    project = db.get_project(project_id)
    bible = get_story_bible(project_id)
    scenes = db.list_scenes(project_id)
    issues = db.list_quality_issues(project_id)
    payload = {"project": project, "story_bible": bible, "scenes": scenes, "quality_issues": issues}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def project_pdf(project_id: str) -> bytes:
    project = db.get_project(project_id)
    bible = get_story_bible(project_id)
    scenes = db.list_scenes(project_id)
    if not project or not scenes:
        raise ValueError("Generate a screenplay before exporting a PDF.")

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter
    margin = 54
    y = height - margin

    def ensure(lines_needed: int = 1, leading: int = 14):
        nonlocal y
        if y < margin + lines_needed * leading:
            pdf.showPage()
            y = height - margin

    def write_wrapped(text: str, font: str = "Courier", size: int = 10, leading: int = 13, indent: int = 0):
        nonlocal y
        pdf.setFont(font, size)
        width = 78 if font.startswith("Courier") else 95
        chunks = textwrap.wrap(text or "", width=width, replace_whitespace=False, drop_whitespace=True) or [""]
        for chunk in chunks:
            ensure(1, leading)
            pdf.drawString(margin + indent, y, chunk)
            y -= leading

    pdf.setTitle(project["title"])
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(letter[0] / 2, height - 72, project["title"])
    y = height - 115
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawCentredString(letter[0] / 2, y, "StoryForge AI — Generated Screenplay")
    y -= 30

    if bible:
        write_wrapped("STORY BIBLE", "Helvetica-Bold", 12, 16)
        write_wrapped(f"Genre: {joined_text(bible.get('genres', []), kind='genre')}", "Helvetica", 9, 12)
        write_wrapped(f"Themes: {joined_text(bible.get('themes', []), kind='theme')}", "Helvetica", 9, 12)
        write_wrapped(f"Summary: {bible.get('summary', '')}", "Helvetica", 9, 12)
        y -= 12

    for scene in scenes:
        ensure(4, 14)
        write_wrapped(scene["slugline"].upper(), "Courier-Bold", 11, 14)
        write_wrapped(scene.get("action", ""), "Courier", 10, 13)
        for dialog in scene.get("dialogue", []):
            ensure(3, 13)
            y -= 5
            pdf.setFont("Courier-Bold", 10)
            pdf.drawCentredString(letter[0] / 2 + 18, y, (dialog.get("character") or "CHARACTER").upper())
            y -= 13
            parenthetical = dialog.get("parenthetical") or ""
            if parenthetical:
                pdf.setFont("Courier", 9)
                pdf.drawCentredString(letter[0] / 2 + 18, y, f"({parenthetical})")
                y -= 12
            for line in textwrap.wrap(dialog.get("text", ""), width=45):
                ensure(1, 13)
                pdf.drawCentredString(letter[0] / 2 + 18, y, line)
                y -= 13
            y -= 4
        for transition in scene.get("transitions", []):
            write_wrapped(transition.upper(), "Courier-Bold", 9, 12)
        y -= 12

    pdf.save()
    buffer.seek(0)
    return buffer.read()
