import asyncio
import os
import struct
import tempfile

import pytest

from rctx.audio_tap import read_frames
from rctx.events import StreamTag


def _encode(tag: int, ts_ms: int, pcm: bytes) -> bytes:
    return struct.pack(">BII", tag, ts_ms, len(pcm)) + pcm


@pytest.mark.asyncio
async def test_read_frames_parses_wire_format():
    """Start a unix-domain server that sends canned frames, and confirm
    ``read_frames`` decodes the wire format correctly.

    We avoid timing-based waits by starting the server first (which blocks
    until the socket file is bound) before kicking off ``read_frames``.
    """
    with tempfile.TemporaryDirectory() as td:
        sock = os.path.join(td, "rctx-test.sock")

        raws = [
            _encode(0, 100, b"\x10\x20\x30\x40"),  # them
            _encode(1, 105, b"\xAA\xBB"),  # me
            _encode(0, 200, b""),  # them, empty payload
        ]

        client_done = asyncio.Event()

        async def handler(_reader, writer):
            try:
                for raw in raws:
                    writer.write(raw)
                    await writer.drain()
                writer.write_eof()
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                client_done.set()

        # start_unix_server returns once the socket is bound and listening —
        # no arbitrary sleep required.
        server = await asyncio.start_unix_server(handler, path=sock)
        try:
            out = []
            async for frame in read_frames(sock):
                out.append(frame)

            # Make sure the server handler has wound down before we exit.
            await asyncio.wait_for(client_done.wait(), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        assert len(out) == 3
        assert out[0].stream_tag == StreamTag.THEM
        assert out[0].timestamp_ms == 100
        assert out[0].pcm == b"\x10\x20\x30\x40"
        assert out[1].stream_tag == StreamTag.ME
        assert out[1].pcm == b"\xAA\xBB"
        assert out[2].pcm == b""
