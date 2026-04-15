#!/usr/bin/env python3
"""Deepgram message-stream diagnostic.

Runs the real rctx audio pipeline (audio-tap -> Deepgram WS) for a fixed
duration and prints every message Deepgram sends to stdout, one per line.
Useful for verifying which message types ("Results", "UtteranceEnd",
"SpeechStarted", "Metadata", etc.) are actually arriving, so we can decide
how to finalize transcripts in the TUI.

Usage:
    DEEPGRAM_API_KEY=... python scripts/dg_diag.py [--duration 30] \
        [--socket-path /tmp/rctx.sock] [--audio-tap PATH]

Talk while it runs. It prints for ``--duration`` seconds then exits cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import websockets

# Make sure we can import rctx when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rctx.audio_tap import read_frames, spawn  # noqa: E402
from rctx.events import StreamTag  # noqa: E402
from rctx.transcribe import DEFAULT_DEEPGRAM_URL  # noqa: E402


def _find_audio_tap_binary() -> Path:
    cand = REPO_ROOT / ".build" / "release" / "AudioTap"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"AudioTap binary not found at {cand} — run `swift build -c release`."
    )


def _ts() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def _fmt_msg(data: dict) -> str:
    mtype = data.get("type", "?")
    if mtype == "Results":
        alts = data.get("channel", {}).get("alternatives", [])
        text = alts[0].get("transcript", "") if alts else ""
        is_final = data.get("is_final")
        speech_final = data.get("speech_final")
        start = data.get("start")
        duration = data.get("duration")
        return (
            f"type=Results is_final={is_final} speech_final={speech_final} "
            f"start={start} dur={duration} text={text!r}"
        )
    if mtype == "UtteranceEnd":
        return (
            f"type=UtteranceEnd last_word_end={data.get('last_word_end')} "
            f"channel={data.get('channel')}"
        )
    if mtype == "SpeechStarted":
        return (
            f"type=SpeechStarted timestamp={data.get('timestamp')} "
            f"channel={data.get('channel')}"
        )
    if mtype == "Metadata":
        return f"type=Metadata request_id={data.get('request_id')}"
    # Fallback: dump full JSON for unknown types.
    return f"type={mtype} raw={json.dumps(data)[:400]}"


async def _send_loop(ws, frames_iter, stop_event: asyncio.Event) -> None:
    try:
        async for frame in frames_iter:
            if stop_event.is_set():
                break
            if frame.stream_tag == StreamTag.THEM and frame.pcm:
                try:
                    await ws.send(frame.pcm)
                except websockets.ConnectionClosed:
                    return
        try:
            await ws.send(json.dumps({"type": "CloseStream"}))
        except websockets.ConnectionClosed:
            pass
    except asyncio.CancelledError:
        raise


async def _recv_loop(ws) -> None:
    async for msg in ws:
        if not isinstance(msg, str):
            print(f"[{_ts()}] <binary {len(msg)} bytes>", flush=True)
            continue
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{_ts()}] <non-json> {msg[:200]!r}", flush=True)
            continue
        print(f"[{_ts()}] {_fmt_msg(data)}", flush=True)


async def _run(duration: float, socket_path: str, binary: Path, url: str) -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("dg_diag: DEEPGRAM_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    print(f"[{_ts()}] spawning audio-tap -> {socket_path}", flush=True)
    proc = await spawn(binary, socket_path)
    stop_event = asyncio.Event()
    try:
        print(f"[{_ts()}] connecting to {url}", flush=True)
        headers = [("Authorization", f"Token {api_key}")]
        async with websockets.connect(url, additional_headers=headers) as ws:
            print(f"[{_ts()}] ws connected; streaming for {duration}s", flush=True)
            frames_iter = read_frames(socket_path)
            send_task = asyncio.create_task(_send_loop(ws, frames_iter, stop_event))
            recv_task = asyncio.create_task(_recv_loop(ws))
            try:
                await asyncio.sleep(duration)
            finally:
                print(f"[{_ts()}] duration elapsed, closing stream", flush=True)
                stop_event.set()
                # Give Deepgram a moment to flush final messages after CloseStream.
                try:
                    await asyncio.wait_for(send_task, timeout=2.0)
                except asyncio.TimeoutError:
                    send_task.cancel()
                try:
                    await asyncio.wait_for(recv_task, timeout=3.0)
                except asyncio.TimeoutError:
                    recv_task.cancel()
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
    finally:
        print(f"[{_ts()}] terminating audio-tap", flush=True)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        print(f"[{_ts()}] done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print every Deepgram message for N seconds.")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to stream (default 30)")
    ap.add_argument("--socket-path", default="/tmp/rctx-diag.sock")
    ap.add_argument("--audio-tap", type=Path, default=None)
    ap.add_argument("--url", default=DEFAULT_DEEPGRAM_URL)
    args = ap.parse_args()

    binary = args.audio_tap or _find_audio_tap_binary()
    try:
        asyncio.run(_run(args.duration, args.socket_path, binary, args.url))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
