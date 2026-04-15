"""Interactive terminal picker for which Claude session to resume."""

from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .session_finder import project_slug


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    mtime: float
    preview: str
    turn_count: int


def list_sessions(project_path: Path) -> list[SessionInfo]:
    """Return Claude sessions for ``project_path`` newest first."""
    home = Path(os.environ.get("HOME", str(Path.home())))
    sessions_dir = home / ".claude" / "projects" / project_slug(project_path)
    if not sessions_dir.exists():
        return []

    out: list[SessionInfo] = []
    for sf in sessions_dir.glob("*.jsonl"):
        try:
            mtime = sf.stat().st_mtime
            lines = sf.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        sid: str | None = None
        first_user_text: str | None = None
        turn_count = 0
        for raw in lines:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if sid is None:
                sid = entry.get("sessionId")
            if entry.get("type") in ("user", "assistant"):
                turn_count += 1
                if first_user_text is None and entry.get("type") == "user":
                    content = entry.get("message", {}).get("content", "")
                    if isinstance(content, str):
                        first_user_text = content
                    elif isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                first_user_text = blk.get("text", "")
                                break
        if not sid:
            continue
        preview = (first_user_text or "(no user messages)").replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        out.append(SessionInfo(session_id=sid, mtime=mtime,
                               preview=preview, turn_count=turn_count))

    out.sort(key=lambda s: s.mtime, reverse=True)
    return out


def prompt_for_session(sessions: list[SessionInfo]) -> str | None:
    """Numbered menu on stderr; read stdin. Return chosen session_id or None."""
    if not sessions:
        return None
    print("", file=sys.stderr)
    print("Available Claude Code sessions:", file=sys.stderr)
    for i, s in enumerate(sessions, start=1):
        ts = datetime.datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {ts}  turns={s.turn_count:3d}  {s.preview}", file=sys.stderr)
    print("  [0] cancel", file=sys.stderr)

    while True:
        try:
            choice = input(f"Pick [1-{len(sessions)}] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice == "":
            return sessions[0].session_id
        if choice == "0":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx].session_id
        except ValueError:
            pass
        print(f"Invalid. 1-{len(sessions)}, or 0 to cancel.", file=sys.stderr)
