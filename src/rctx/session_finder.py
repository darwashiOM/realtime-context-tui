"""Locate the most-recent Claude Code session .jsonl for a project."""

from __future__ import annotations

import json
import os
from pathlib import Path


def project_slug(project_path: Path) -> str:
    """Replicate Claude Code's slug naming: abs path with / -> -"""
    return str(project_path.resolve()).replace("/", "-")


def find_most_recent_session(project_path: Path) -> str | None:
    """Return the sessionId of the most-recently-modified .jsonl, or None."""
    home = Path(os.environ.get("HOME", str(Path.home())))
    sessions_dir = home / ".claude" / "projects" / project_slug(project_path)
    if not sessions_dir.exists():
        return None

    candidates = sorted(sessions_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for session_file in candidates:
        try:
            first = session_file.read_text(errors="ignore").splitlines()[:1]
            if not first:
                continue
            entry = json.loads(first[0])
            sid = entry.get("sessionId")
            if isinstance(sid, str) and sid:
                return sid
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    return None
