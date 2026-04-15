import json
import time
from pathlib import Path

import pytest

from rctx.session_finder import find_most_recent_session, project_slug


def test_project_slug_replaces_slashes():
    assert project_slug(Path("/Users/me/code/proj")) == "-Users-me-code-proj"


def test_find_most_recent_session_returns_newest_file_session_id(tmp_path, monkeypatch):
    proj = tmp_path / "me" / "proj"
    proj.mkdir(parents=True)
    slug = project_slug(proj)
    sessions = tmp_path / ".claude" / "projects" / slug
    sessions.mkdir(parents=True)

    older = sessions / "older.jsonl"
    older.write_text(json.dumps({"sessionId": "old-uuid", "type": "user"}) + "\n")
    time.sleep(0.05)
    newer = sessions / "newer.jsonl"
    newer.write_text(json.dumps({"sessionId": "new-uuid", "type": "user"}) + "\n")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_most_recent_session(proj) == "new-uuid"


def test_find_most_recent_session_returns_none_when_no_sessions(tmp_path, monkeypatch):
    proj = tmp_path / "fresh"
    proj.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_most_recent_session(proj) is None
