"""Long-lived `claude --resume` subprocess wrapper: QuestionEvent -> ResponseChunk."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence

from .events import Citation, QuestionEvent, ResponseChunk
from .retriever import RetrievalHit

CLAUDE_MODEL = "claude-sonnet-4-6"


def _build_user_turn(q: QuestionEvent, hits: Sequence[RetrievalHit]) -> str:
    """Compose the text content sent as the user turn."""
    lines = [
        "A meeting participant just asked me this question. Answer in my voice, "
        "first person, 2-4 sentences, citing code as file:line_start-line_end when relevant.",
        "",
        f"Question: {q.text}",
    ]
    if hits:
        lines.append("")
        lines.append("Fresh code chunks the retriever surfaced (may or may not be relevant):")
        for h in hits:
            lines.append(f"--- {h.file}:{h.line_start}-{h.line_end} ---")
            lines.append(h.snippet)
    return "\n".join(lines)


def _parse_line(raw: bytes) -> tuple[str | None, bool]:
    """Parse one stream-json stdout line.

    Returns (text_delta_or_none, is_end_of_response).

    Schema captured from `claude` v2.1.109 on 2026-04-15 via
    scripts/claude_stream_diag.py:
      - {"type":"system","subtype":"init",...}       -> ignored
      - {"type":"assistant",
         "message":{"content":[{"type":"text","text":"..."}]}} -> text delta
      - {"type":"rate_limit_event",...}              -> ignored
      - {"type":"result","subtype":"success",...}    -> end-of-response
    Without --include-partial-messages, assistant messages arrive whole rather
    than as content_block_delta chunks. We still handle both shapes below so
    the parser keeps working if that flag is ever turned on, or if a future
    Claude Code version switches to delta-only streaming.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, False

    kind = obj.get("type")

    # End-of-response signals.
    if kind in ("result", "message_stop"):
        return None, True

    # Shape (a): full assistant message with a list of content blocks.
    if kind == "assistant":
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list):
            text_parts = [
                blk.get("text", "") for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            joined = "".join(text_parts)
            if joined:
                return joined, False

    # Shape (b): incremental text_delta event (when --include-partial-messages).
    if kind == "content_block_delta":
        delta = obj.get("delta", {})
        text = delta.get("text", "")
        if text:
            return text, False

    return None, False


class Answerer:
    """Owns a long-lived `claude --resume` subprocess, serializes questions."""

    def __init__(
        self,
        *,
        session_id: str,
        model: str = CLAUDE_MODEL,
        _spawn_override: Optional[Callable[[], Awaitable]] = None,
    ) -> None:
        self._session_id = session_id
        self._model = model
        self._proc = None
        self._lock = asyncio.Lock()
        self._spawn_override = _spawn_override

    async def start(self) -> None:
        if self._spawn_override is not None:
            self._proc = await self._spawn_override()
            return
        # --verbose is required by the CLI when output-format=stream-json.
        self._proc = await asyncio.create_subprocess_exec(
            "claude",
            "--resume", self._session_id,
            "--print",
            "--verbose",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--model", self._model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            await self._proc.wait()

    async def answer(
        self,
        question: QuestionEvent,
        hits: Sequence[RetrievalHit],
        *,
        question_id: int,
    ) -> AsyncIterator[ResponseChunk]:
        if self._proc is None:
            raise RuntimeError("Answerer not started")

        async with self._lock:
            user_turn = _build_user_turn(question, hits)
            payload = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": user_turn},
            }) + "\n"
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()

            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                delta, is_end = _parse_line(line)
                if delta:
                    yield ResponseChunk(
                        question_id=question_id,
                        text_delta=delta,
                        is_final=False,
                    )
                if is_end:
                    break

            citations = tuple(
                Citation(file=h.file, line_start=h.line_start, line_end=h.line_end)
                for h in hits
            )
            yield ResponseChunk(
                question_id=question_id,
                text_delta="",
                is_final=True,
                citations=citations,
            )
