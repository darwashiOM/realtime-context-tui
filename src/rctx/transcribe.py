"""Stream audio frames to Deepgram and yield UtteranceEvents."""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

import websockets

from .events import Frame, StreamTag, UtteranceEvent

DEFAULT_DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&language=en-US"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&interim_results=true"
    "&endpointing=300"
    "&utterance_end_ms=1000"
    "&vad_events=true"
)


class DeepgramAuthError(RuntimeError):
    """Raised when no API key is available."""


async def run(
    frames: AsyncIterator[Frame],
    *,
    url: str | None = None,
    api_key: str | None = None,
    stream_filter: StreamTag = StreamTag.THEM,
) -> AsyncIterator[UtteranceEvent]:
    """Stream frames matching ``stream_filter`` to Deepgram; yield events in arrival order.

    Defaults to ``StreamTag.THEM``. Pass ``stream_filter=StreamTag.ME`` to
    transcribe your own mic instead (e.g. for the self-coach pipeline).

    The mock-test server uses a path-only URL; production uses the full
    DEFAULT_DEEPGRAM_URL with all params. ``api_key`` falls back to
    DEEPGRAM_API_KEY env var.
    """
    url = url or DEFAULT_DEEPGRAM_URL
    api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise DeepgramAuthError("DEEPGRAM_API_KEY env var not set")

    headers = [("Authorization", f"Token {api_key}")]

    async with websockets.connect(url, additional_headers=headers) as ws:
        out_queue: asyncio.Queue[UtteranceEvent | None] = asyncio.Queue()

        async def send_loop() -> None:
            try:
                async for frame in frames:
                    if frame.stream_tag == stream_filter and frame.pcm:
                        await ws.send(frame.pcm)
                # Tell Deepgram we're done so it can emit any final transcript.
                try:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except websockets.ConnectionClosed:
                    pass
            except asyncio.CancelledError:
                raise

        async def recv_loop() -> None:
            try:
                async for msg in ws:
                    if not isinstance(msg, str):
                        continue
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") != "Results":
                        # We could synthesize on UtteranceEnd events too, but
                        # speech_final on Results is sufficient for this plan.
                        continue
                    alts = data.get("channel", {}).get("alternatives", [])
                    if not alts:
                        continue
                    text = alts[0].get("transcript", "")
                    if not text:
                        continue
                    start = float(data.get("start", 0.0))
                    duration = float(data.get("duration", 0.0))
                    await out_queue.put(
                        UtteranceEvent(
                            text=text,
                            is_final=bool(data.get("is_final", False)),
                            speech_final=bool(data.get("speech_final", False)),
                            start_ms=int(start * 1000),
                            end_ms=int((start + duration) * 1000),
                        )
                    )
            finally:
                await out_queue.put(None)

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())

        try:
            while True:
                ev = await out_queue.get()
                if ev is None:
                    return
                yield ev
        finally:
            send_task.cancel()
            recv_task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)
