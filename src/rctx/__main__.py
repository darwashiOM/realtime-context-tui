"""rctx CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .orchestrator import run as run_orchestrator
from .session_finder import find_most_recent_session
from .session_picker import list_sessions, prompt_for_session


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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--socket-path", default="/tmp/rctx.sock")
    parser.add_argument("--audio-tap", type=Path, default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--custom-instruction", default=None,
                        help="Preamble for every answer. If omitted, prompt for one.")
    parser.add_argument("--no-picker", action="store_true",
                        help="Skip interactive picker; use most-recent session.")
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("rctx: DEEPGRAM_API_KEY not set.", file=sys.stderr)
        return 2
    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    # Session selection
    sid = args.session_id
    if sid is None:
        if args.no_picker:
            sid = find_most_recent_session(args.project)
        else:
            sessions = list_sessions(args.project)
            if not sessions:
                print(f"rctx: no Claude sessions for {args.project}. "
                      "Start one with `claude` in that directory first.", file=sys.stderr)
                return 2
            sid = prompt_for_session(sessions)
    if not sid:
        print("rctx: no session chosen, exiting.", file=sys.stderr)
        return 2

    # Custom instruction prompt
    instruction = args.custom_instruction
    if instruction is None:
        print("\nOptional session-level instruction for Claude "
              "(empty to skip, e.g. 'be extra concise'):", file=sys.stderr)
        try:
            instruction = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            instruction = ""

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
                session_id=sid,
                custom_instruction=instruction or "",
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
