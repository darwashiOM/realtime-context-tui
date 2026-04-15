# Self-Coach for Live Speech (Plan 5)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute. Steps use checkbox (`- [ ]`) tracking.

**Goal:** Add a live coach pane that listens to **your own** mic, and after each finalized utterance asks Claude to suggest the **next thing you should say** — to correct factual mistakes, fill in things you left out, or move the explanation forward. **Not** a rewrite of what you said.

**Non-regression:** Existing flows (system audio → transcript → question detection → answer in Q+A pane) stay untouched. This is purely additive.

**Architecture additions:**
- `transcribe.run()` gains a `stream_filter: StreamTag` kwarg (default `THEM`).
- Orchestrator opens a **second** socket connection to `audio-tap` and runs a second `transcribe.run(stream_filter=ME)` for your mic.
- New `CoachChunk` event type.
- `Answerer.coach()` method serialized through the same `_lock` (so coach + answer share one `claude --resume` subprocess; no risk of two writers racing on the session `.jsonl`).
- UI:
  - Existing transcript pane shows `[me]` prefix (dim italic) on your own finalized utterances, alongside their utterances.
  - New `RichLog#coach` pane below Q+A (magenta border), streams coach suggestions.
- Coach prompt asks for the NEXT utterance, not a rewrite — explicitly distinct from the answerer's question-answer behavior.

**Working directory:** `/Users/darwashi/Downloads/interview/realtime-context-tui` (on `main`).

**Pre-flight:** `uv run pytest -v` → 30 tests pass.

---

### Task 1: `transcribe.run` accepts `stream_filter`

**Files:**
- Modify: `src/rctx/transcribe.py`
- Modify: `Tests/test_transcribe.py`

- [ ] **Step 1: Add `stream_filter` kwarg to `run()`**

In `src/rctx/transcribe.py`, change the `run` signature:

```python
async def run(
    frames: AsyncIterator[Frame],
    *,
    url: str | None = None,
    api_key: str | None = None,
    stream_filter: StreamTag = StreamTag.THEM,
) -> AsyncIterator[UtteranceEvent]:
```

In the `send_loop` body, replace `if frame.stream_tag == StreamTag.THEM` with:
```python
if frame.stream_tag == stream_filter and frame.pcm:
```

- [ ] **Step 2: Update `Tests/test_transcribe.py`**

Add a test that exercises `stream_filter=StreamTag.ME` — exactly mirrors the existing test, but assert that 'me' frames are forwarded and 'them' frames are dropped.

```python
@pytest.mark.asyncio
async def test_transcribe_filters_to_me_when_requested():
    received_pcm = bytearray()

    async def mock_dg(ws):
        async def reader():
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        received_pcm.extend(msg)
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        await ws.send(json.dumps({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "ok"}]},
            "is_final": True, "speech_final": True,
            "start": 0.0, "duration": 0.5,
        }))
        await asyncio.sleep(0.1)
        await ws.close()
        reader_task.cancel()

    server = await websockets.serve(mock_dg, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    mock_url = f"ws://127.0.0.1:{port}/v1/listen"

    async def frames():
        yield Frame(stream_tag=StreamTag.THEM, timestamp_ms=0, pcm=b"\x00\x01" * 320)
        yield Frame(stream_tag=StreamTag.ME, timestamp_ms=20, pcm=b"\x02\x03" * 320)
        yield Frame(stream_tag=StreamTag.ME, timestamp_ms=40, pcm=b"\x04\x05" * 320)

    out = []
    async for ev in transcribe_run(frames(), url=mock_url, api_key="dummy",
                                    stream_filter=StreamTag.ME):
        out.append(ev)

    server.close()
    await server.wait_closed()

    # Only 2 ME frames × 640 bytes = 1280; THEM frame must be skipped.
    assert len(received_pcm) == 2 * 640
    assert any(e.text == "ok" for e in out)
```

- [ ] **Step 3: Run, verify all transcribe tests pass**

Run: `uv run pytest Tests/test_transcribe.py -v`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rctx/transcribe.py Tests/test_transcribe.py
git commit -m "feat(rctx): transcribe accepts stream_filter (default THEM)"
```

---

### Task 2: Add `CoachChunk` event

**Files:**
- Modify: `src/rctx/events.py`
- Modify: `Tests/test_events.py`

- [ ] **Step 1: Append to `src/rctx/events.py`**

```python
@dataclass(frozen=True, slots=True)
class CoachChunk:
    """A streamed piece of a coach suggestion (next-utterance hint).

    ``coach_id`` ties chunks to the speech turn they suggest a continuation
    for. ``is_final`` marks the last chunk.
    """

    coach_id: int
    text_delta: str
    is_final: bool
```

- [ ] **Step 2: Add a test in `Tests/test_events.py`**

```python
def test_coach_chunk_fields():
    from rctx.events import CoachChunk

    c = CoachChunk(coach_id=3, text_delta="Try saying ", is_final=False)
    assert c.coach_id == 3
    assert c.text_delta == "Try saying "
    assert c.is_final is False
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest Tests/test_events.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/rctx/events.py Tests/test_events.py
git commit -m "feat(rctx): CoachChunk event type"
```

---

### Task 3: `Answerer.coach` method

**Files:**
- Modify: `src/rctx/answerer.py`
- Modify: `Tests/test_answerer.py`

**Purpose:** Add a `coach(my_recent_speech: str, *, coach_id: int)` method to `Answerer` that uses the existing `claude --resume` subprocess (serialized through the lock), with a different prompt asking for the next-utterance suggestion. Yields `CoachChunk` events.

- [ ] **Step 1: Add a coach prompt builder + method to `src/rctx/answerer.py`**

After `_build_user_turn`, add:

```python
def _build_coach_turn(my_recent_speech: str, custom_instruction: str = "") -> str:
    lines: list[str] = []
    if custom_instruction:
        lines.append(
            f"Session-level instruction from the presenter "
            f"(follow this for every suggestion): {custom_instruction}"
        )
        lines.append("")
    lines += [
        "I'm presenting a project right now. The text below is what I just "
        "said out loud. Based on our prior project conversation, suggest the "
        "NEXT thing I should say to do one or more of:",
        "- Correct a factual mistake I just made",
        "- Clarify something I left out",
        "- Move the explanation forward",
        "",
        "Do NOT rewrite what I just said. Suggest what comes next.",
        "",
        "Format EXACTLY:",
        "",
        "**Next:** <one or two sentences I should say next, in my voice, "
        "confident and conversational. No file paths, no jargon I haven't "
        "introduced.>",
        "",
        "If — and only if — I just said something factually wrong, also include:",
        "**Note:** <one sentence flagging the mistake so I know to correct it>",
        "",
        "Do not add any other sections.",
        "",
        f"What I just said: {my_recent_speech}",
    ]
    return "\n".join(lines)
```

Then add this method to `Answerer`:

```python
async def coach(
    self,
    my_recent_speech: str,
    *,
    coach_id: int,
) -> AsyncIterator[CoachChunk]:
    """Stream a coach suggestion for what to say next."""
    if self._proc is None:
        raise RuntimeError("Answerer not started")

    async with self._lock:
        payload = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": _build_coach_turn(my_recent_speech, self._custom_instruction),
            },
        }) + "\n"
        self._proc.stdin.write(payload.encode())
        await self._proc.stdin.drain()

        saw_delta = False
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            delta, source, is_end = _parse_line(line)
            if delta and source == "delta":
                saw_delta = True
                yield CoachChunk(coach_id=coach_id, text_delta=delta, is_final=False)
            elif delta and source == "whole" and not saw_delta:
                yield CoachChunk(coach_id=coach_id, text_delta=delta, is_final=False)
            if is_end:
                break

        yield CoachChunk(coach_id=coach_id, text_delta="", is_final=True)
```

Make sure to import `CoachChunk` at the top of the file.

> NOTE: Match the exact `_parse_line` return shape that's currently in `answerer.py`. The reference above assumes `(delta, source, is_end)`. If the file's existing `_parse_line` returns a different tuple, mirror what `answer()` does — copy that loop's structure.

- [ ] **Step 2: Add a test to `Tests/test_answerer.py`**

```python
@pytest.mark.asyncio
async def test_answerer_coach_streams_chunks_with_correct_prompt():
    lines = [
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "**Next:** Try "}}}).encode() + b"\n",
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "saying it this way."}}}).encode() + b"\n",
        json.dumps({"type": "result", "subtype": "success"}).encode() + b"\n",
    ]
    fake = _FakeProc(lines)

    async def fake_spawn():
        return fake

    ans = Answerer(session_id="test", _spawn_override=fake_spawn)
    await ans.start()

    chunks = []
    async for c in ans.coach("um the resampler does some math i think", coach_id=42):
        chunks.append(c)

    await ans.stop()

    assert chunks[-1].is_final is True
    assert all(c.coach_id == 42 for c in chunks)
    body = "".join(c.text_delta for c in chunks if not c.is_final)
    assert "**Next:**" in body or "Try saying" in body

    sent = json.loads(fake.stdin.written[0])
    assert "What I just said" in sent["message"]["content"]
    assert "rewrite" in sent["message"]["content"].lower()  # the coach rule
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest Tests/test_answerer.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/rctx/answerer.py Tests/test_answerer.py
git commit -m "feat(rctx): Answerer.coach for next-utterance suggestions"
```

---

### Task 4: UI — coach pane + `[me]` prefix in transcript

**Files:**
- Modify: `src/rctx/ui.py`
- Modify: `Tests/test_ui.py`

- [ ] **Step 1: Add new pane and methods**

In `src/rctx/ui.py`:

1. Update `CSS` to add a coach pane. Adjust transcript/Q+A heights.

```python
CSS = """
Screen { layout: vertical; }
RichLog#transcript { height: 2fr; border: solid green; padding: 0 1; }
Static#interim { height: auto; min-height: 1; padding: 0 1; color: $text-muted; }
RichLog#qa { height: 2fr; border: solid cyan; padding: 0 1; }
RichLog#coach { height: 2fr; border: solid magenta; padding: 0 1; }
Static#status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
"""
```

2. Update `compose()` to yield a `RichLog(id="coach", markup=True, highlight=False, wrap=True)` between the Q+A RichLog and the status Static.

3. Add `__init__` shadow state:

```python
self._coach_lines: list[str] = []
self._current_coach_buf: dict[int, str] = {}
```

4. Add new public methods (place near `on_response_chunk`):

```python
def on_my_utterance(self, event: UtteranceEvent) -> None:
    """Render YOUR finalized utterances into the transcript with a [me] prefix."""
    if event.is_final:
        line = f"[dim italic][me][/dim italic] {event.text}"
        self.query_one("#transcript", RichLog).write(line)
        self._transcript_lines.append(line)

def on_coach_started(self, coach_id: int) -> None:
    coach = self.query_one("#coach", RichLog)
    sep = f"[dim]── coach #{coach_id} ──[/dim]"
    coach.write(sep)
    self._coach_lines.append(sep)
    self._current_coach_buf[coach_id] = ""

def on_coach_chunk(self, chunk: 'CoachChunk') -> None:
    coach = self.query_one("#coach", RichLog)
    if chunk.is_final:
        coach.write("")
        self._coach_lines.append("")
        self._current_coach_buf.pop(chunk.coach_id, None)
        return
    self._current_coach_buf[chunk.coach_id] = (
        self._current_coach_buf.get(chunk.coach_id, "") + chunk.text_delta
    )
    coach.write(chunk.text_delta)
    if (self._coach_lines
            and not self._coach_lines[-1].startswith("[dim]── coach")
            and self._coach_lines[-1] != ""):
        self._coach_lines[-1] += chunk.text_delta
    else:
        self._coach_lines.append(chunk.text_delta)
```

(Import `CoachChunk` at top.)

5. Add test helper:

```python
def coach_text(self) -> str:
    return "\n".join(self._coach_lines)
```

- [ ] **Step 2: Add tests to `Tests/test_ui.py`**

```python
@pytest.mark.asyncio
async def test_my_utterance_lands_in_transcript_with_prefix():
    from rctx.events import UtteranceEvent
    app = TranscribeApp()
    async with app.run_test() as pilot:
        utt = UtteranceEvent(text="okay so the resampler does some math",
                             is_final=True, speech_final=False, start_ms=0, end_ms=1000)
        app.on_my_utterance(utt)
        await pilot.pause()
        assert "[me]" in app.transcript_text()
        assert "the resampler does some math" in app.transcript_text()


@pytest.mark.asyncio
async def test_coach_pane_shows_streamed_suggestion():
    from rctx.events import CoachChunk
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.on_coach_started(coach_id=1)
        app.on_coach_chunk(CoachChunk(coach_id=1, text_delta="**Next:** ", is_final=False))
        app.on_coach_chunk(CoachChunk(coach_id=1, text_delta="Try saying it like this.", is_final=False))
        app.on_coach_chunk(CoachChunk(coach_id=1, text_delta="", is_final=True))
        await pilot.pause()
        text = app.coach_text()
        assert "coach #1" in text
        assert "Try saying it like this." in text
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest Tests/test_ui.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/rctx/ui.py Tests/test_ui.py
git commit -m "feat(rctx): UI coach pane + [me] prefix in transcript"
```

---

### Task 5: Orchestrator — second transcriber + coach pipeline

**Files:**
- Modify: `src/rctx/orchestrator.py`

- [ ] **Step 1: Replace `src/rctx/orchestrator.py`** with the version below

```python
"""Wire audio-tap → 2× transcribe → (transcript + classifier→answer + coach)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

from .answerer import Answerer
from .audio_tap import read_frames, spawn
from .classifier import is_question
from .events import QuestionEvent, StreamTag, UtteranceEvent
from .retriever import Retriever
from .transcribe import run as transcribe_run
from .ui import TranscribeApp


async def run(
    audio_tap_binary: Path,
    socket_path: str,
    project_path: Path,
    session_id: str,
    custom_instruction: str = "",
) -> None:
    app = TranscribeApp()
    qa_counter = itertools.count(1)
    coach_counter = itertools.count(1)

    async def them_pump(retriever: Retriever, answerer: Answerer) -> None:
        frames = read_frames(socket_path)
        async for ev in transcribe_run(frames, stream_filter=StreamTag.THEM):
            app.append_event(ev)
            if ev.is_final and is_question(ev):
                asyncio.create_task(_handle_question(ev, retriever, answerer, app, qa_counter))

    async def me_pump(answerer: Answerer) -> None:
        frames = read_frames(socket_path)
        async for ev in transcribe_run(frames, stream_filter=StreamTag.ME):
            if ev.is_final:
                app.on_my_utterance(ev)
                # Fire-and-forget coach
                asyncio.create_task(_handle_my_utterance(ev, answerer, app, coach_counter))

    async def pump() -> None:
        app.set_status(f"indexing {project_path}…")
        retriever = Retriever(project_path=project_path)
        retriever.build()

        app.set_status(f"starting claude --resume {session_id[:8]}…")
        answerer = Answerer(session_id=session_id, custom_instruction=custom_instruction)
        await answerer.start()

        app.set_status(f"spawning audio-tap → {socket_path}")
        proc = await spawn(audio_tap_binary, socket_path)
        try:
            app.set_status(f"ready. session={session_id[:8]}…")
            them_task = asyncio.create_task(them_pump(retriever, answerer))
            me_task = asyncio.create_task(me_pump(answerer))
            await asyncio.gather(them_task, me_task)
        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            await answerer.stop()

    pump_task = asyncio.create_task(pump())
    try:
        await app.run_async()
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass


async def _handle_question(
    utterance: UtteranceEvent,
    retriever: Retriever,
    answerer: Answerer,
    app: TranscribeApp,
    counter: itertools.count,
) -> None:
    try:
        q = QuestionEvent(text=utterance.text, source_utterance=utterance)
        qid = next(counter)
        app.on_question_detected(q, question_id=qid)
        hits = retriever.search(q.text, k=5)
        async for chunk in answerer.answer(q, hits, question_id=qid):
            app.on_response_chunk(chunk)
    except Exception as exc:
        app.set_status(f"answer error: {exc!r}")


async def _handle_my_utterance(
    utterance: UtteranceEvent,
    answerer: Answerer,
    app: TranscribeApp,
    counter: itertools.count,
) -> None:
    try:
        cid = next(counter)
        app.on_coach_started(cid)
        async for chunk in answerer.coach(utterance.text, coach_id=cid):
            app.on_coach_chunk(chunk)
    except Exception as exc:
        app.set_status(f"coach error: {exc!r}")
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 3: Smoke CLI**

Run: `uv run rctx --help` — should still work.

- [ ] **Step 4: Commit + push**

```bash
git add src/rctx/orchestrator.py
git commit -m "feat(rctx): orchestrator runs ME pipeline alongside THEM with coach"
git push origin main
```

---

### Task 6: End-to-end smoke test (manual)

- [ ] **Step 1: Launch**

```bash
uv run rctx --project /Users/darwashi/Downloads/interview/realtime-context-tui
```

Pick a session, optional instruction.

- [ ] **Step 2: Talk into the mic** (no system audio playing — isolate ME stream)

Say something messy on purpose: *"so the resampler does some math, um, I think it might use, uh, AVConverter or something to do the resampling at like 48 to 16."*

Expected:
- Transcript pane shows the line with `[me]` prefix.
- Coach pane shows `── coach #1 ──` then streams a `**Next:**` suggestion that doesn't repeat what you said but extends or corrects (e.g. *"Specifically, it uses Apple's AVAudioConverter and the conversion runs on a per-format cache so we don't re-allocate when the input format changes."*) — possibly followed by a `**Note:**` if you said something wrong.

- [ ] **Step 3: Play system audio asking a question** (test non-regression)

Play audio of someone asking *"how does the resampler handle 48 kHz?"*.

Expected: Q+A pane fires as before. Coach pane unaffected. They+coach run in parallel.

- [ ] **Step 4: Quit & confirm**

`q`. Done criterion: both pipelines fired without interfering with each other.

---

## Done criteria for Plan 5

- `uv run pytest -v` → all tests pass (≥34 total).
- Speaking into the mic produces a `[me]`-prefixed line in Transcript AND a streaming coach suggestion in the Coach pane.
- Question detection from system audio still works in Q+A pane (no regression).
- All commits on `main`, pushed.
