# Orchestrator + Deepgram + TUI Implementation Plan (Plan 2)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute. Steps use checkbox (`- [ ]`) tracking.

**Goal:** A working live-transcription tool. `rctx` spawns the `audio-tap` binary, consumes its socket, streams the system-audio channel to Deepgram, and renders live transcripts in a `textual` TUI. No classifier, no LLM, no retrieval yet — those land in Plan 3.

**Architecture:** Python `asyncio` orchestrator. Three async producers/consumers wired through `asyncio.Queue`:
- `audio_tap.read_frames(socket)` → yields `Frame` events
- `transcribe.run(frames_in, deepgram_url, api_key)` → consumes `them`-tagged frames, yields `UtteranceEvent`s from Deepgram
- `ui.TranscribeApp` → consumes `UtteranceEvent`s and renders them in panes

Audio-tap subprocess is spawned by `rctx` and torn down on shutdown.

**Tech stack:** Python 3.12+, `uv` (env mgr), `websockets` (Deepgram WS client), `textual` (TUI), `pytest` + `pytest-asyncio` (tests). No `deepgram-sdk` — using raw WebSocket gives us tighter control and easier mocking.

**Wire format consumed (set by Plan 1's audio-tap):**
```
[1 byte stream tag: 0x00=them, 0x01=me]
[4 bytes BE uint32 timestamp_ms]
[4 bytes BE uint32 payload_len]
[N bytes Int16 LE 16kHz mono PCM]
```

**Working directory:** `/Users/darwashi/Downloads/interview/realtime-context-tui` (work on `main`).

**Pre-flight:**
- `python3 --version` → 3.12+
- `uv --version` → installed (else: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `echo "${DEEPGRAM_API_KEY:+set}"` → `set`
- `.build/release/AudioTap` exists from Plan 1.

---

### Task 1: Python project init

**Files:**
- Create: `pyproject.toml`
- Create: `src/rctx/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "rctx"
version = "0.0.1"
description = "Realtime context TUI: live audio transcription + RAG over Claude Code history"
requires-python = ">=3.12"
dependencies = [
    "websockets>=13.0",
    "textual>=0.80.0",
]

[project.scripts]
rctx = "rctx.__main__:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rctx"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 110
target-version = "py312"
```

- [ ] **Step 2: Write `src/rctx/__init__.py`**

```python
"""rctx — realtime context TUI."""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write `tests/__init__.py`** — empty file.

- [ ] **Step 4: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures for rctx tests."""
```

- [ ] **Step 5: Sync deps and verify**

Run: `uv sync && uv run pytest -q`
Expected: deps install, pytest finds 0 tests but exits 0.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/rctx/ tests/
git commit -m "feat(rctx): python project scaffold (uv + textual + websockets)"
```

---

### Task 2: events.py (TDD)

**Files:**
- Create: `src/rctx/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write failing test in `tests/test_events.py`**

```python
from rctx.events import Frame, StreamTag, UtteranceEvent


def test_stream_tag_values():
    assert StreamTag.THEM == 0
    assert StreamTag.ME == 1


def test_frame_is_frozen_dataclass():
    f = Frame(stream_tag=StreamTag.THEM, timestamp_ms=100, pcm=b"\x01\x02")
    assert f.stream_tag == StreamTag.THEM
    assert f.timestamp_ms == 100
    assert f.pcm == b"\x01\x02"
    import dataclasses
    assert dataclasses.is_dataclass(f)


def test_utterance_event_fields():
    u = UtteranceEvent(
        text="hello world",
        is_final=True,
        speech_final=False,
        start_ms=1000,
        end_ms=2500,
    )
    assert u.text == "hello world"
    assert u.is_final is True
    assert u.speech_final is False
    assert u.end_ms - u.start_ms == 1500
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_events.py -v`
Expected: ImportError — `events` doesn't exist yet.

- [ ] **Step 3: Implement `src/rctx/events.py`**

```python
"""Typed events flowing through the rctx pipeline."""

from dataclasses import dataclass
from enum import IntEnum


class StreamTag(IntEnum):
    THEM = 0
    ME = 1


@dataclass(frozen=True, slots=True)
class Frame:
    """One audio frame from audio-tap. PCM is 16kHz mono Int16 little-endian."""

    stream_tag: StreamTag
    timestamp_ms: int
    pcm: bytes


@dataclass(frozen=True, slots=True)
class UtteranceEvent:
    """One transcription event emitted by the transcriber.

    ``is_final`` marks the last interim result for a clause (won't be amended).
    ``speech_final`` marks the end of a complete utterance (speaker stopped).
    """

    text: str
    is_final: bool
    speech_final: bool
    start_ms: int
    end_ms: int
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/events.py tests/test_events.py
git commit -m "feat(rctx): typed events (Frame, UtteranceEvent, StreamTag)"
```

---

### Task 3: audio_tap.py — socket reader + subprocess (TDD)

**Files:**
- Create: `src/rctx/audio_tap.py`
- Create: `tests/test_audio_tap.py`

**Purpose:** `read_frames(socket_path)` async iterator yields `Frame`s. `spawn(binary_path, socket_path)` launches the Swift binary as a subprocess with proper lifecycle.

- [ ] **Step 1: Write failing tests in `tests/test_audio_tap.py`**

```python
import asyncio
import os
import struct
import tempfile

import pytest

from rctx.audio_tap import read_frames
from rctx.events import StreamTag


async def _serve_one_client(socket_path: str, frames_to_send: list[bytes]) -> None:
    """Tiny test server: accept one client, write the supplied raw frame bytes, close."""

    async def handler(_reader, writer):
        for raw in frames_to_send:
            writer.write(raw)
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handler, path=socket_path)
    async with server:
        await server.wait_closed()


def _encode(tag: int, ts_ms: int, pcm: bytes) -> bytes:
    return struct.pack(">BII", tag, ts_ms, len(pcm)) + pcm


@pytest.mark.asyncio
async def test_read_frames_parses_wire_format():
    with tempfile.TemporaryDirectory() as td:
        sock = os.path.join(td, "rctx-test.sock")

        raws = [
            _encode(0, 100, b"\x10\x20\x30\x40"),  # them
            _encode(1, 105, b"\xAA\xBB"),  # me
            _encode(0, 200, b""),  # them, empty payload
        ]

        server_task = asyncio.create_task(_serve_one_client(sock, raws))
        # Give the server a moment to bind.
        await asyncio.sleep(0.05)

        out = []
        async for frame in read_frames(sock):
            out.append(frame)

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        assert len(out) == 3
        assert out[0].stream_tag == StreamTag.THEM
        assert out[0].timestamp_ms == 100
        assert out[0].pcm == b"\x10\x20\x30\x40"
        assert out[1].stream_tag == StreamTag.ME
        assert out[1].pcm == b"\xAA\xBB"
        assert out[2].pcm == b""
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_audio_tap.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/rctx/audio_tap.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_audio_tap.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/audio_tap.py tests/test_audio_tap.py
git commit -m "feat(rctx): audio-tap socket reader + subprocess spawn"
```

---

### Task 4: transcribe.py — Deepgram streaming (TDD with mock WS)

**Files:**
- Create: `src/rctx/transcribe.py`
- Create: `tests/test_transcribe.py`

**Purpose:** Connect to Deepgram's streaming WS, send `THEM` PCM as binary frames, parse JSON results into `UtteranceEvent`s. Test against a local mock WS server so no real key needed in CI.

**Deepgram URL params:**
- `model=nova-3`, `language=en-US`
- `encoding=linear16`, `sample_rate=16000`, `channels=1`
- `interim_results=true`, `endpointing=300`, `utterance_end_ms=1000`, `vad_events=true`

- [ ] **Step 1: Write failing test in `tests/test_transcribe.py`**

```python
import asyncio
import json
from typing import AsyncIterator

import pytest
import websockets

from rctx.events import Frame, StreamTag, UtteranceEvent
from rctx.transcribe import run as transcribe_run


@pytest.mark.asyncio
async def test_transcribe_emits_events_from_mock_deepgram():
    received_pcm = bytearray()

    async def mock_dg(ws):
        # Server-side: read a few frames of binary PCM, then push canned JSON.
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
        await asyncio.sleep(0.05)
        await ws.close()
        reader_task.cancel()

    server = await websockets.serve(mock_dg, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    mock_url = f"ws://127.0.0.1:{port}/v1/listen"

    async def frames() -> AsyncIterator[Frame]:
        for ts in (0, 50, 100):
            yield Frame(stream_tag=StreamTag.THEM, timestamp_ms=ts, pcm=b"\x00\x01" * 320)
        # also send a `me` frame — should be ignored
        yield Frame(stream_tag=StreamTag.ME, timestamp_ms=120, pcm=b"\xff" * 320)

    out: list[UtteranceEvent] = []
    async for ev in transcribe_run(frames(), url=mock_url, api_key="dummy"):
        out.append(ev)

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
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/rctx/transcribe.py`**

```python
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
) -> AsyncIterator[UtteranceEvent]:
    """Stream ``THEM`` frames to Deepgram; yield events in arrival order.

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
                    if frame.stream_tag == StreamTag.THEM and frame.pcm:
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
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/transcribe.py tests/test_transcribe.py
git commit -m "feat(rctx): Deepgram streaming transcriber with mock-WS test"
```

---

### Task 5: ui.py — TUI shell (TDD with textual.Pilot)

**Files:**
- Create: `src/rctx/ui.py`
- Create: `tests/test_ui.py`

**Purpose:** Three-pane TUI using `textual`. Top: Transcript (rolling). Middle: "Detected" pane (placeholder for Plan 3, shows current is_final clause). Bottom: status bar.

- [ ] **Step 1: Write failing test in `tests/test_ui.py`**

```python
import pytest

from rctx.ui import TranscribeApp
from rctx.events import UtteranceEvent


@pytest.mark.asyncio
async def test_app_renders_interim_and_final_transcript_lines():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.append_event(UtteranceEvent(
            text="hello", is_final=False, speech_final=False,
            start_ms=0, end_ms=500,
        ))
        app.append_event(UtteranceEvent(
            text="hello world", is_final=True, speech_final=True,
            start_ms=0, end_ms=1200,
        ))
        await pilot.pause()
        # RichLog stores written lines; assert content surfaces.
        log_text = app.transcript_text()
        assert "hello world" in log_text


@pytest.mark.asyncio
async def test_app_set_status_updates_status_bar():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.set_status("READY")
        await pilot.pause()
        assert app.status_text() == "READY"
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_ui.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/rctx/ui.py`**

```python
"""Textual TUI for live transcription."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, RichLog, Static

from .events import UtteranceEvent


class TranscribeApp(App):
    """Three-pane live-transcription view: header + transcript + status."""

    CSS = """
    Screen { layout: vertical; }
    RichLog#transcript { height: 1fr; border: solid green; padding: 0 1; }
    Static#status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", markup=True, highlight=False, wrap=True)
        yield Static("starting…", id="status")

    # --- Public API used by orchestrator ---

    def append_event(self, event: UtteranceEvent) -> None:
        log = self.query_one("#transcript", RichLog)
        if event.is_final:
            log.write(event.text)
        else:
            log.write(f"[dim]{event.text}[/dim]")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # --- Test helpers ---

    def transcript_text(self) -> str:
        log = self.query_one("#transcript", RichLog)
        return "\n".join(str(line) for line in log.lines)

    def status_text(self) -> str:
        return str(self.query_one("#status", Static).renderable)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_ui.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/ui.py tests/test_ui.py
git commit -m "feat(rctx): textual TUI with transcript + status panes"
```

---

### Task 6: orchestrator.py + __main__.py — wire everything

**Files:**
- Create: `src/rctx/orchestrator.py`
- Create: `src/rctx/__main__.py`

**Purpose:** CLI entry point. Spawns `audio-tap`, reads frames, fans them to transcribe, pumps events into the TUI.

- [ ] **Step 1: Write `src/rctx/orchestrator.py`**

```python
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
```

- [ ] **Step 2: Write `src/rctx/__main__.py`**

```python
"""rctx CLI entry point: ``rctx --project <path>``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .orchestrator import run as run_orchestrator


def _find_audio_tap_binary() -> Path:
    """Locate the audio-tap binary built by Plan 1."""
    here = Path(__file__).resolve()
    # Walk up to repo root (which has .build/release/AudioTap).
    for ancestor in here.parents:
        cand = ancestor / ".build" / "release" / "AudioTap"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find .build/release/AudioTap — did you run "
        "`swift build -c release` in the repo root?"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="rctx", description="Realtime context TUI.")
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Project directory whose Claude Code transcripts + source we'll RAG over.",
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/rctx.sock",
        help="Unix socket where audio-tap will publish PCM frames.",
    )
    parser.add_argument(
        "--audio-tap",
        type=Path,
        default=None,
        help="Path to the audio-tap binary (auto-detected if omitted).",
    )
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print(
            "rctx: DEEPGRAM_API_KEY is not set. "
            "Run `export DEEPGRAM_API_KEY=...` in your shell rc.",
            file=sys.stderr,
        )
        return 2

    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify CLI loads and rejects bad input**

Run: `uv run rctx --help`
Expected: argparse help text printed.

Run: `uv run rctx --project /nonexistent`
Expected: Exits 2 with `--project path does not exist` error (assumes `DEEPGRAM_API_KEY` is set; otherwise the missing-key check fires first — verify with `env -u DEEPGRAM_API_KEY uv run rctx --project /tmp` if needed).

- [ ] **Step 4: Commit**

```bash
git add src/rctx/orchestrator.py src/rctx/__main__.py
git commit -m "feat(rctx): orchestrator + CLI wiring everything together"
```

---

### Task 7: End-to-end smoke test (manual)

No file changes — operator test.

- [ ] **Step 1: Run the full pipeline**

In one terminal (in repo root):

```sh
uv run rctx --project ~/  --socket-path /tmp/rctx-e2e.sock
```

(`--project ~/` is just a placeholder existing path; the project arg isn't used by Plan 2's UI yet.)

Expected: A textual TUI appears with header, empty transcript pane, and status bar showing `connected. project=/Users/darwashi`.

- [ ] **Step 2: Speak through speakers**

Play any audio with English speech (YouTube, podcast, etc.). Within ~1–3 s, transcribed text should start appearing in the transcript pane. Interim results show dimmed; finalized results show bright.

- [ ] **Step 3: Quit cleanly**

Press `q` (or Ctrl-C). The TUI should exit, audio-tap subprocess should terminate, and `/tmp/rctx-e2e.sock` should be removed (audio-tap unlinks on its own SIGINT path).

- [ ] **Step 4: Run the unit tests once more**

Run: `uv run pytest -v`
Expected: All 7 tests pass (events: 3, audio_tap: 1, transcribe: 1, ui: 2).

- [ ] **Step 5: Push**

```bash
git push origin main
```

---

## Done criteria for Plan 2

- `uv sync` installs deps cleanly.
- `uv run pytest` reports all tests passing.
- `uv run rctx --project <any-path>` shows the TUI; speaking into mic / playing audio produces live transcripts in the transcript pane.
- Quit via `q` or Ctrl-C cleanly tears down the subprocess.
- All commits on `main`, pushed.

On finishing: hand off to Plan 3 (classifier + retriever + answerer + speculation + whisper fallback).
