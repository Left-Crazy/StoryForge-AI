from .. import db


def create(title: str, description: str = "") -> dict:
    if not title.strip():
        raise ValueError("Project title is required.")
    return db.create_project(title.strip(), description.strip())


def list_all() -> list[dict]:
    return db.list_projects()


def delete(project_id: str) -> None:
    db.delete_project(project_id)
