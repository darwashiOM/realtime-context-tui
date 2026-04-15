"""rctx CLI entry point: ``rctx --project <path>``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .orchestrator import run as run_orchestrator


def _find_audio_tap_binary() -> Path:
    """Locate the audio-tap binary built by Plan 1."""
    here = Path(__file__).resolve()
    # Walk up to repo root (which has .build/release/AudioTap).
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
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Project directory whose Claude Code transcripts + source we'll RAG over.",
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/rctx.sock",
        help="Unix socket where audio-tap will publish PCM frames.",
    )
    parser.add_argument(
        "--audio-tap",
        type=Path,
        default=None,
        help="Path to the audio-tap binary (auto-detected if omitted).",
    )
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print(
            "rctx: DEEPGRAM_API_KEY is not set. "
            "Run `export DEEPGRAM_API_KEY=...` in your shell rc.",
            file=sys.stderr,
        )
        return 2

    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
