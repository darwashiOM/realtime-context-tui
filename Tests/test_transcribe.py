import asyncio
import json
from typing import AsyncIterator

import pytest
import websockets

from rctx.events import Frame, StreamTag, UtteranceEvent
from rctx.transcribe import run as transcribe_run


@pytest.mark.asyncio
async def test_transcribe_emits_events_from_mock_deepgram():
    """Spin up a tiny WS server that mimics Deepgram's Results frames and
    verify:
      - THEM PCM frames are forwarded as binary
      - ME frames are NOT forwarded
      - JSON Results events become UtteranceEvents with correct mapping
    """
    received_pcm = bytearray()
    server_done = asyncio.Event()

    async def mock_dg(ws):
        # Receive binary PCM while pushing canned JSON back.
        async def reader():
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray, memoryview)):
                        received_pcm.extend(bytes(msg))
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(reader())
        try:
            # Push two Results frames spaced out so the client definitely
            # has time to finish its send loop too.
            await ws.send(json.dumps({
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "hello"}]},
                "is_final": False,
                "speech_final": False,
                "start": 0.0,
                "duration": 0.5,
            }))
            await ws.send(json.dumps({
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "hello world"}]},
                "is_final": True,
                "speech_final": True,
                "start": 0.0,
                "duration": 1.2,
            }))
            # Give the client side a moment to drain its sends before we
            # close — we want received_pcm to have all THEM frames.
            await asyncio.sleep(0.2)
        finally:
            await ws.close()
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            server_done.set()

    server = await websockets.serve(mock_dg, "127.0.0.1", 0)
    try:
        port = next(iter(server.sockets)).getsockname()[1]
        mock_url = f"ws://127.0.0.1:{port}/v1/listen"

        async def frames() -> AsyncIterator[Frame]:
            for ts in (0, 50, 100):
                yield Frame(stream_tag=StreamTag.THEM, timestamp_ms=ts, pcm=b"\x00\x01" * 320)
            # also send a `me` frame — should be ignored
            yield Frame(stream_tag=StreamTag.ME, timestamp_ms=120, pcm=b"\xff" * 320)

        out: list[UtteranceEvent] = []
        async for ev in transcribe_run(frames(), url=mock_url, api_key="dummy"):
            out.append(ev)

        await asyncio.wait_for(server_done.wait(), timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()

    assert len(out) == 2
    assert out[0].text == "hello"
    assert out[0].is_final is False
    assert out[1].text == "hello world"
    assert out[1].is_final is True
    assert out[1].speech_final is True
    # 'me' frame must NOT have been forwarded; only the 3 'them' frames.
    assert len(received_pcm) == 3 * 640


@pytest.mark.asyncio
async def test_transcribe_filters_to_me_when_requested():
    """With stream_filter=StreamTag.ME, only ME frames are forwarded and
    THEM frames are dropped — the inverse of the default."""
    received_pcm = bytearray()
    server_done = asyncio.Event()

    async def mock_dg(ws):
        async def reader():
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray, memoryview)):
                        received_pcm.extend(bytes(msg))
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(reader())
        try:
            await ws.send(json.dumps({
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "ok"}]},
                "is_final": True,
                "speech_final": True,
                "start": 0.0,
                "duration": 0.5,
            }))
            await asyncio.sleep(0.2)
        finally:
            await ws.close()
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            server_done.set()

    server = await websockets.serve(mock_dg, "127.0.0.1", 0)
    try:
        port = next(iter(server.sockets)).getsockname()[1]
        mock_url = f"ws://127.0.0.1:{port}/v1/listen"

        async def frames() -> AsyncIterator[Frame]:
            yield Frame(stream_tag=StreamTag.THEM, timestamp_ms=0, pcm=b"\x00\x01" * 320)
            yield Frame(stream_tag=StreamTag.ME, timestamp_ms=20, pcm=b"\x02\x03" * 320)
            yield Frame(stream_tag=StreamTag.ME, timestamp_ms=40, pcm=b"\x04\x05" * 320)

        out: list[UtteranceEvent] = []
        async for ev in transcribe_run(
            frames(), url=mock_url, api_key="dummy", stream_filter=StreamTag.ME
        ):
            out.append(ev)

        await asyncio.wait_for(server_done.wait(), timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()

    # Only 2 ME frames × 640 bytes = 1280; THEM frame must be skipped.
    assert len(received_pcm) == 2 * 640
    assert any(e.text == "ok" for e in out)
