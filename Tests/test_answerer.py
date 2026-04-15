import asyncio
import json
from typing import AsyncIterator

import pytest

from rctx.answerer import Answerer
from rctx.events import QuestionEvent, UtteranceEvent
from rctx.retriever import RetrievalHit


class _FakeStream:
    """Mimics asyncio.StreamReader/Writer for unit testing."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self.written: list[bytes] = []

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeProc:
    def __init__(self, out_lines: list[bytes]):
        self.stdin = _FakeStream([])
        self.stdout = _FakeStream(out_lines)
        self.stderr = _FakeStream([])
        self._returncode: int | None = None

    def terminate(self) -> None:
        self._returncode = 143

    async def wait(self) -> int:
        return self._returncode or 0


@pytest.mark.asyncio
async def test_answerer_streams_deltas_and_emits_final():
    """Whole-message fallback path (no --include-partial-messages).

    Schema captured from `claude --resume ... --print --verbose
    --input-format stream-json --output-format stream-json` on v2.1.109:
      {"type":"system","subtype":"init",...}         -> ignored
      {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}} -> text
      {"type":"rate_limit_event",...}                -> ignored
      {"type":"result","subtype":"success",...}      -> end-of-response
    """
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}).encode() + b"\n",
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Re"}]}}).encode() + b"\n",
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "sampling uses AVAudioConverter."}]}}).encode() + b"\n",
        json.dumps({"type": "rate_limit_event", "rate_limit_info": {}}).encode() + b"\n",
        json.dumps({"type": "result", "subtype": "success"}).encode() + b"\n",
    ]
    fake = _FakeProc(lines)

    async def fake_spawn():
        return fake

    ans = Answerer(session_id="test-session-id", _spawn_override=fake_spawn)
    await ans.start()

    utt = UtteranceEvent(text="how does resample work?", is_final=True,
                         speech_final=False, start_ms=0, end_ms=1000)
    q = QuestionEvent(text=utt.text, source_utterance=utt)
    hits = [RetrievalHit(file="src/rctx/retriever.py", line_start=1, line_end=10,
                         snippet="# snippet", score=1.5)]

    chunks = []
    async for chunk in ans.answer(q, hits, question_id=7):
        chunks.append(chunk)

    await ans.stop()

    # Expect at least one text_delta + a final chunk
    assert any(c.text_delta for c in chunks)
    assert chunks[-1].is_final is True
    assert chunks[-1].question_id == 7
    # Citations are carried on the final chunk
    assert chunks[-1].citations[0].file == "src/rctx/retriever.py"
    # The user turn was sent to stdin as a JSON line
    assert len(fake.stdin.written) == 1
    sent = json.loads(fake.stdin.written[0])
    assert sent["type"] == "user"
    assert "how does resample work" in sent["message"]["content"]


@pytest.mark.asyncio
async def test_answerer_streams_partial_message_deltas():
    """With --include-partial-messages, claude v2.1.109 emits content_block_delta
    wrapped in a stream_event envelope. The trailing whole `assistant` message
    must be suppressed to avoid double-counting the text."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "message_start",
                              "message": {"role": "assistant", "content": []}}}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "content_block_start", "index": 0,
                              "content_block": {"type": "text", "text": ""}}}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": "Re"}}}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta",
                                        "text": "sampling uses AVAudioConverter."}}}).encode() + b"\n",
        # Whole assistant message comes too; must be deduped.
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text",
                                              "text": "Resampling uses AVAudioConverter."}]}}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "content_block_stop", "index": 0}}).encode() + b"\n",
        json.dumps({"type": "stream_event",
                    "event": {"type": "message_stop"}}).encode() + b"\n",
        json.dumps({"type": "result", "subtype": "success"}).encode() + b"\n",
    ]
    fake = _FakeProc(lines)

    async def fake_spawn():
        return fake

    ans = Answerer(session_id="test-session-id", _spawn_override=fake_spawn)
    await ans.start()

    utt = UtteranceEvent(text="how does resample work?", is_final=True,
                         speech_final=False, start_ms=0, end_ms=1000)
    q = QuestionEvent(text=utt.text, source_utterance=utt)
    hits: list[RetrievalHit] = []

    chunks = []
    async for chunk in ans.answer(q, hits, question_id=11):
        chunks.append(chunk)

    await ans.stop()

    # Text chunks in order. Deltas fire as they arrive - responsive UX.
    text_chunks = [c.text_delta for c in chunks if c.text_delta]
    assert text_chunks == ["Re", "sampling uses AVAudioConverter."], (
        f"Expected two delta chunks in order; got: {text_chunks}. "
        f"If a third chunk appeared, the whole-message dedup broke."
    )
    # Final chunk still emitted with empty text_delta and correct id.
    assert chunks[-1].is_final is True
    assert chunks[-1].question_id == 11


@pytest.mark.asyncio
async def test_answerer_prepends_custom_instruction_to_user_turn():
    lines = [b'{"type":"result","subtype":"success"}\n']
    fake = _FakeProc(lines)

    async def fake_spawn():
        return fake

    ans = Answerer(
        session_id="test",
        _spawn_override=fake_spawn,
        custom_instruction="be extra concise",
    )
    await ans.start()
    utt = UtteranceEvent(text="how does x work?", is_final=True,
                         speech_final=False, start_ms=0, end_ms=500)
    q = QuestionEvent(text=utt.text, source_utterance=utt)

    async for _ in ans.answer(q, [], question_id=1):
        pass

    await ans.stop()
    sent = json.loads(fake.stdin.written[0])
    assert "be extra concise" in sent["message"]["content"]
