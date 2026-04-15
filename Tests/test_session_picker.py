import io
import json
import time
from pathlib import Path

import pytest

from rctx.session_picker import (
    SessionInfo,
    list_sessions,
    prompt_for_session,
)
from rctx.session_finder import project_slug


def _write_session(sessions_dir: Path, sid: str, first_user: str, turns: int) -> Path:
    f = sessions_dir / f"{sid}.jsonl"
    lines = [json.dumps({"sessionId": sid, "type": "user",
                         "message": {"content": first_user}})]
    for i in range(turns - 1):
        lines.append(json.dumps({"sessionId": sid, "type": "assistant",
                                 "message": {"content": [{"type": "text", "text": f"a{i}"}]}}))
    f.write_text("\n".join(lines))
    return f


def test_list_sessions_sorted_newest_first(tmp_path, monkeypatch):
    proj = tmp_path / "me" / "proj"
    proj.mkdir(parents=True)
    sdir = tmp_path / ".claude" / "projects" / project_slug(proj)
    sdir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    _write_session(sdir, "old", "older question", turns=3)
    time.sleep(0.05)
    _write_session(sdir, "new", "newer question really long text that should truncate", turns=5)

    out = list_sessions(proj)
    assert [s.session_id for s in out] == ["new", "old"]
    assert out[0].turn_count == 5
    assert "newer question" in out[0].preview
    assert len(out[0].preview) <= 80


def test_list_sessions_empty_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert list_sessions(tmp_path / "nonexistent") == []


def test_prompt_for_session_returns_chosen_id(monkeypatch, capsys):
    sessions = [
        SessionInfo(session_id="s1", mtime=1.0, preview="p1", turn_count=3),
        SessionInfo(session_id="s2", mtime=2.0, preview="p2", turn_count=6),
    ]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    chosen = prompt_for_session(sessions)
    assert chosen == "s2"


def test_prompt_for_session_default_on_empty(monkeypatch):
    sessions = [SessionInfo(session_id="a", mtime=0.0, preview="p", turn_count=1)]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert prompt_for_session(sessions) == "a"


def test_prompt_for_session_cancel_returns_none(monkeypatch):
    sessions = [SessionInfo(session_id="a", mtime=0.0, preview="p", turn_count=1)]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "0")
    assert prompt_for_session(sessions) is None
