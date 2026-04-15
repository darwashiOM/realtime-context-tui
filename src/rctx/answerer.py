"""Long-lived `claude --resume` subprocess wrapper: QuestionEvent -> ResponseChunk."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence

from .events import Citation, QuestionEvent, ResponseChunk
from .retriever import RetrievalHit

CLAUDE_MODEL = "claude-sonnet-4-6"


def _build_user_turn(
    q: QuestionEvent,
    hits: Sequence[RetrievalHit],
    *,
    custom_instruction: str = "",
) -> str:
    """Compose the text content sent as the user turn."""
    lines: list[str] = []
    if custom_instruction:
        lines.append(
            f"Session-level instruction from the presenter "
            f"(follow this for every question): {custom_instruction}"
        )
        lines.append("")
    lines += [
        "A meeting participant just asked me this question. Answer in my voice, "
        "first person. Format the answer EXACTLY like this:",
        "",
        "**Short answer:** <one or two plain-English sentences I can say aloud "
        "right now to the person asking. No jargon, no file names, no line numbers.>",
        "",
        "**Where:**",
        "- `path/file.ext:line_start-line_end` \u2014 <3-7 word description>",
        "- `path/file.ext:line_start-line_end` \u2014 <3-7 word description>",
        "",
        "Rules:",
        "- Include 2-4 entries under **Where:**; pick the most relevant.",
        "- Keep each description to 3-7 words.",
        "- Do not add any other sections, headings, or prose outside this format.",
        "- Do not wrap the whole answer in a code block.",
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


def _parse_line(raw: bytes) -> tuple[str | None, bool, str | None]:
    """Parse one stream-json stdout line.

    Returns (text_or_none, is_end_of_response, source).
    `source` is "delta" for streaming partials, "whole" for a full
    assistant message, or None for non-text / control lines. Callers use
    `source` to dedupe: with --include-partial-messages, both shapes arrive
    for the same turn and the whole-message line must be dropped.

    Schema captured from `claude` v2.1.109 on 2026-04-15 via
    scripts/claude_stream_diag.py. With --include-partial-messages:
      - {"type":"system","subtype":"init",...}       -> ignored
      - {"type":"stream_event","event":{"type":"content_block_delta",
         "delta":{"type":"text_delta","text":"..."}}} -> text delta
      - {"type":"stream_event","event":{"type":"message_stop"}} -> ignored
         (result follows; we end on that to avoid cutting off early)
      - {"type":"assistant","message":{...}}         -> whole message, skipped
         (would double-count the text we already streamed)
      - {"type":"rate_limit_event",...}              -> ignored
      - {"type":"result","subtype":"success",...}    -> end-of-response

    Without --include-partial-messages, no stream_event lines arrive and the
    whole `assistant` message carries the text. We still accept that shape
    as a fallback so we're resilient if the flag name changes or gets
    dropped in a future Claude Code version.

    The Answerer loop tracks whether any stream_event deltas arrived during
    this response; if so, the trailing whole-message `assistant` line is
    suppressed by the loop (not here). This function returns whatever the
    line contains; dedup is the caller's job.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, False, None

    kind = obj.get("type")

    # Unwrap the stream_event envelope used by --include-partial-messages.
    if kind == "stream_event":
        ev = obj.get("event", {}) or {}
        ev_kind = ev.get("type")
        if ev_kind == "content_block_delta":
            delta = ev.get("delta", {}) or {}
            text = delta.get("text", "")
            if text:
                return text, False, "delta"
        # message_start / content_block_start / content_block_stop /
        # message_delta / message_stop carry no user-visible text and we
        # prefer to end the response on the top-level `result` line.
        return None, False, None

    # End-of-response signals.
    if kind in ("result", "message_stop"):
        return None, True, None

    # Bare content_block_delta (historical / non-wrapped variant).
    if kind == "content_block_delta":
        delta = obj.get("delta", {}) or {}
        text = delta.get("text", "")
        if text:
            return text, False, "delta"

    # Whole assistant message with a list of content blocks. Only useful
    # when partials are off; the Answerer loop suppresses this if it has
    # already seen stream_event deltas for the current turn.
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
                return joined, False, "whole"

    return None, False, None


class Answerer:
    """Owns a long-lived `claude --resume` subprocess, serializes questions."""

    def __init__(
        self,
        *,
        session_id: str,
        model: str = CLAUDE_MODEL,
        custom_instruction: str = "",
        _spawn_override: Optional[Callable[[], Awaitable]] = None,
    ) -> None:
        self._session_id = session_id
        self._model = model
        self._custom_instruction = custom_instruction
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
            "--include-partial-messages",
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
            user_turn = _build_user_turn(
                question, hits, custom_instruction=self._custom_instruction
            )
            payload = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": user_turn},
            }) + "\n"
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()

            saw_delta = False
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                text, is_end, source = _parse_line(line)
                if text and source == "delta":
                    saw_delta = True
                    yield ResponseChunk(
                        question_id=question_id,
                        text_delta=text,
                        is_final=False,
                    )
                elif text and source == "whole" and not saw_delta:
                    # Fallback when --include-partial-messages isn't active:
                    # no stream_event deltas arrived, so emit the whole
                    # assistant message as a single chunk.
                    yield ResponseChunk(
                        question_id=question_id,
                        text_delta=text,
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
