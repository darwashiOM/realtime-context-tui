"""Wire together audio-tap → transcribe → UI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .audio_tap import read_frames, spawn
from .transcribe import run as transcribe_run
from .ui import TranscribeApp


async def run(audio_tap_binary: Path, socket_path: str, project_path: Path) -> None:
    """Top-level coroutine that owns process + sockets + UI lifecycle."""
    app = TranscribeApp()

    async def pump() -> None:
        # 1) Spawn audio-tap and wait for socket to appear.
        app.set_status(f"spawning audio-tap → {socket_path}")
        proc = await spawn(audio_tap_binary, socket_path)
        try:
            app.set_status(f"connected. project={project_path}")
            # 2) Read frames and pipe to transcribe; pipe events to UI.
            frames = read_frames(socket_path)
            try:
                async for event in transcribe_run(frames):
                    app.append_event(event)
            except Exception as exc:
                app.set_status(f"transcribe error: {exc!r}")
        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    pump_task = asyncio.create_task(pump())
    try:
        await app.run_async()
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
