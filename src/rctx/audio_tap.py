"""Spawn the audio-tap subprocess and consume its Unix-socket frame stream."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from typing import AsyncIterator

from .events import Frame, StreamTag

_HEADER_FMT = ">BII"
_HEADER_LEN = 9


async def read_frames(socket_path: str) -> AsyncIterator[Frame]:
    """Connect to ``socket_path`` and yield ``Frame``s as they arrive.

    Closes cleanly on EOF or cancellation.
    """
    reader, writer = await asyncio.open_unix_connection(path=socket_path)
    try:
        while True:
            try:
                header = await reader.readexactly(_HEADER_LEN)
            except asyncio.IncompleteReadError:
                return
            tag, ts_ms, payload_len = struct.unpack(_HEADER_FMT, header)
            payload = (
                await reader.readexactly(payload_len) if payload_len > 0 else b""
            )
            yield Frame(
                stream_tag=StreamTag(tag),
                timestamp_ms=ts_ms,
                pcm=payload,
            )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def spawn(binary_path: Path, socket_path: str) -> asyncio.subprocess.Process:
    """Launch ``audio-tap`` and wait until its socket file appears.

    Caller is responsible for terminate/wait on the returned process.
    """
    # Clean any stale socket file so we can detect when audio-tap creates a fresh one.
    sock = Path(socket_path)
    if sock.exists():
        sock.unlink()

    proc = await asyncio.create_subprocess_exec(
        str(binary_path),
        "--socket-path",
        socket_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait up to 5 s for the binary to bind its socket.
    for _ in range(50):
        if sock.exists():
            return proc
        await asyncio.sleep(0.1)

    proc.terminate()
    await proc.wait()
    raise RuntimeError(
        f"audio-tap did not create socket at {socket_path} within 5s"
    )
