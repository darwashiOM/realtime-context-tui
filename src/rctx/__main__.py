"""rctx CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .orchestrator import run as run_orchestrator
from .session_finder import find_most_recent_session


def _find_audio_tap_binary() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        cand = ancestor / ".build" / "release" / "AudioTap"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find .build/release/AudioTap — did you run "
        "`swift build -c release` in the repo root?"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="rctx", description="Realtime context TUI.")
    parser.add_argument("--project", type=Path, required=True,
                        help="Project dir (its Claude Code session will be resumed).")
    parser.add_argument("--socket-path", default="/tmp/rctx.sock")
    parser.add_argument("--audio-tap", type=Path, default=None)
    parser.add_argument("--session-id", default=None,
                        help="Override auto-detected Claude session ID.")
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("rctx: DEEPGRAM_API_KEY not set.", file=sys.stderr)
        return 2
    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    sid = args.session_id or find_most_recent_session(args.project)
    if not sid:
        print(
            f"rctx: no Claude session found for {args.project}. "
            "Start one with `claude` in that directory first, or pass --session-id.",
            file=sys.stderr,
        )
        return 2

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
                session_id=sid,
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
