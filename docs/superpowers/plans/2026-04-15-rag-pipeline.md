# RAG Pipeline Implementation Plan (Plan 3)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute. Steps use checkbox (`- [ ]`) tracking.

**Goal:** Add question detection + grounded answers to `rctx`. Questions are surfaced to a Q+A pane with streaming responses from a persistent Claude Code subprocess (`claude --resume <session-id>`). **No API keys required** — uses the user's existing Claude Code subscription.

**Architecture:**
- **Classifier** — local regex heuristic (interrogative words + `?` + question-verb patterns). Zero latency, zero cost.
- **Session finder** — at startup, locate the most recent `.jsonl` under `~/.claude/projects/<slug>/` and extract its `sessionId` field. User can override via `--session-id`.
- **Retriever** — ripgrep + BM25 over project source files for fresh code citations. Indexed once at startup.
- **Answerer** — wraps a long-lived subprocess: `claude --resume <id> --print --input-format stream-json --output-format stream-json --model claude-sonnet-4-6`. Sends each question as a JSON user message on stdin, parses streaming assistant deltas from stdout, yields `ResponseChunk` events.
- **UI** — gains Q+A pane (questions in cyan, streamed answers, citation footer).
- **Orchestrator** — fires classifier-then-retriever-then-answer for every `is_final` utterance; non-questions ignored.

**Why subprocess over SDK:** The resumed Claude Code session already contains the entire build history (design docs, Plan 1+2 conversations, code changes). Its on-disk prompt cache is pre-warmed. No need to re-send 80k tokens of context per question.

**Tech stack additions:** `rank-bm25>=0.2.2` only. No `anthropic` SDK.

**Working directory:** `/Users/darwashi/Downloads/interview/realtime-context-tui` (on `main`).

**Pre-flight:**
- `claude --version` → 2.x or newer (Claude Code CLI installed)
- `claude --help | grep -E 'resume|input-format|output-format'` → shows those flags
- `ls ~/.claude/projects/-Users-darwashi-Downloads-interview-realtime-context-tui/*.jsonl` → at least one session file exists (this conversation)
- All Plan 2 tests still green: `uv run pytest -v`

---

### Task 1: Add new event types (QuestionEvent, Citation, ResponseChunk)

**Files:**
- Modify: `src/rctx/events.py`
- Modify: `Tests/test_events.py`

- [ ] **Step 1: Append to `src/rctx/events.py`** (after existing dataclasses):

```python
@dataclass(frozen=True, slots=True)
class QuestionEvent:
    """A finalized utterance classified as a question to the presenter."""

    text: str
    source_utterance: UtteranceEvent


@dataclass(frozen=True, slots=True)
class Citation:
    """A file:line reference shown in the UI below an answer."""

    file: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ResponseChunk:
    """A streamed piece of an answer.

    ``question_id`` ties chunks to the question they answer. ``is_final``
    marks the last chunk (carries the final ``citations`` tuple).
    """

    question_id: int
    text_delta: str
    is_final: bool
    citations: tuple[Citation, ...] = ()
```

- [ ] **Step 2: Extend `Tests/test_events.py`**:

```python
from rctx.events import Citation, QuestionEvent, ResponseChunk, UtteranceEvent


def test_question_event_wraps_utterance():
    u = UtteranceEvent(text="how does x work", is_final=True, speech_final=False,
                       start_ms=0, end_ms=1000)
    q = QuestionEvent(text="how does x work", source_utterance=u)
    assert q.text == "how does x work"
    assert q.source_utterance is u


def test_response_chunk_fields():
    c = Citation(file="src/foo.py", line_start=10, line_end=12)
    rc = ResponseChunk(question_id=1, text_delta="It works by ", is_final=False, citations=(c,))
    assert rc.question_id == 1
    assert rc.text_delta == "It works by "
    assert rc.citations[0].file == "src/foo.py"
```

- [ ] **Step 3: Run, verify pass**

Run: `uv run pytest Tests/test_events.py -v`
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rctx/events.py Tests/test_events.py
git commit -m "feat(rctx): QuestionEvent, Citation, ResponseChunk events"
```

---

### Task 2: session_finder.py — locate the Claude session to resume

**Files:**
- Create: `src/rctx/session_finder.py`
- Create: `Tests/test_session_finder.py`

**Purpose:** Given a project path, locate the most-recently-modified `.jsonl` under `~/.claude/projects/<slug>/` and extract its `sessionId`.

Claude Code `.jsonl` entries typically contain `{"sessionId":"<uuid>", ...}` on every line. We grab the sessionId from the first line of the most recent file.

- [ ] **Step 1: Write failing test `Tests/test_session_finder.py`**

```python
import json
import time
from pathlib import Path

import pytest

from rctx.session_finder import find_most_recent_session, project_slug


def test_project_slug_replaces_slashes():
    assert project_slug(Path("/Users/me/code/proj")) == "-Users-me-code-proj"


def test_find_most_recent_session_returns_newest_file_session_id(tmp_path, monkeypatch):
    proj = tmp_path / "me" / "proj"
    proj.mkdir(parents=True)
    slug = project_slug(proj)
    sessions = tmp_path / ".claude" / "projects" / slug
    sessions.mkdir(parents=True)

    older = sessions / "older.jsonl"
    older.write_text(json.dumps({"sessionId": "old-uuid", "type": "user"}) + "\n")
    time.sleep(0.05)
    newer = sessions / "newer.jsonl"
    newer.write_text(json.dumps({"sessionId": "new-uuid", "type": "user"}) + "\n")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_most_recent_session(proj) == "new-uuid"


def test_find_most_recent_session_returns_none_when_no_sessions(tmp_path, monkeypatch):
    proj = tmp_path / "fresh"
    proj.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_most_recent_session(proj) is None
```

- [ ] **Step 2: Run, verify fail (ImportError)**

- [ ] **Step 3: Implement `src/rctx/session_finder.py`**

```python
"""Locate the most-recent Claude Code session .jsonl for a project."""

from __future__ import annotations

import json
import os
from pathlib import Path


def project_slug(project_path: Path) -> str:
    """Replicate Claude Code's slug naming: abs path with / → -"""
    return str(project_path.resolve()).replace("/", "-")


def find_most_recent_session(project_path: Path) -> str | None:
    """Return the sessionId of the most-recently-modified .jsonl, or None."""
    home = Path(os.environ.get("HOME", str(Path.home())))
    sessions_dir = home / ".claude" / "projects" / project_slug(project_path)
    if not sessions_dir.exists():
        return None

    candidates = sorted(sessions_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for session_file in candidates:
        try:
            first = session_file.read_text(errors="ignore").splitlines()[:1]
            if not first:
                continue
            entry = json.loads(first[0])
            sid = entry.get("sessionId")
            if isinstance(sid, str) and sid:
                return sid
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    return None
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest Tests/test_session_finder.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/session_finder.py Tests/test_session_finder.py
git commit -m "feat(rctx): find most-recent Claude session for a project path"
```

---

### Task 3: classifier.py — local regex question detector

**Files:**
- Create: `src/rctx/classifier.py`
- Create: `Tests/test_classifier.py`

**Purpose:** Decide whether a finalized utterance is a question. Local regex; no network; biased toward "yes" (cheap false positives, fatal false negatives).

- [ ] **Step 1: Write failing test `Tests/test_classifier.py`**

```python
from rctx.classifier import is_question
from rctx.events import UtteranceEvent


def _u(text: str, is_final: bool = True) -> UtteranceEvent:
    return UtteranceEvent(text=text, is_final=is_final, speech_final=False,
                          start_ms=0, end_ms=1000)


def test_interrogative_phrases_flagged():
    assert is_question(_u("how does this work"))
    assert is_question(_u("why is it slow"))
    assert is_question(_u("what does the classifier do"))
    assert is_question(_u("can you explain that again"))
    assert is_question(_u("is this production ready"))
    assert is_question(_u("are you sure"))
    assert is_question(_u("could you walk me through it"))


def test_trailing_question_mark_flagged():
    assert is_question(_u("ok so the flow is clear?"))


def test_statements_not_flagged():
    assert not is_question(_u("yeah that makes sense"))
    assert not is_question(_u("cool"))
    assert not is_question(_u("got it thanks"))
    assert not is_question(_u("interesting"))


def test_interim_utterances_not_flagged():
    assert not is_question(_u("how does", is_final=False))


def test_empty_or_whitespace_not_flagged():
    assert not is_question(_u(""))
    assert not is_question(_u("   "))
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/rctx/classifier.py`**

```python
"""Local regex-based question classifier. Generous-triggering by design."""

from __future__ import annotations

import re

from .events import UtteranceEvent

# Start-of-sentence interrogatives (case-insensitive, must be the first word
# or follow ". "/"? "). Keep this list broad; false positives are cheap.
_INTERROGATIVE_STARTS = re.compile(
    r"(?:^|[.?!]\s+)(how|why|what|when|where|which|who|"
    r"can|could|would|should|will|do|does|did|is|are|am|was|were|"
    r"may|might|have|has|had)\b",
    re.IGNORECASE,
)


def is_question(utterance: UtteranceEvent) -> bool:
    """Return True if the utterance looks like a question to the presenter."""
    if not utterance.is_final:
        return False
    text = utterance.text.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    if _INTERROGATIVE_STARTS.search(text):
        return True
    return False
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest Tests/test_classifier.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/classifier.py Tests/test_classifier.py
git commit -m "feat(rctx): local regex question classifier"
```

---

### Task 4: retriever.py — BM25 over project source

**Files:**
- Create: `src/rctx/retriever.py`
- Create: `Tests/test_retriever.py`

**Purpose:** Walk the project at startup, chunk files into ~50-line blocks, build a BM25 index. On demand, return top-k `RetrievalHit`s with `file:line` citations.

- [ ] **Step 1: Add `rank-bm25` dependency**

```bash
uv add "rank-bm25>=0.2.2"
```

- [ ] **Step 2: Write failing test `Tests/test_retriever.py`**

```python
from pathlib import Path

from rctx.retriever import Retriever


def test_retriever_indexes_code_files_and_returns_relevant_chunk(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "audio.py").write_text(
        "def resample(buf):\n"
        "    # resample 48khz float32 to 16khz int16\n"
        "    return converted_bytes\n"
    )
    (tmp_path / "src" / "unrelated.py").write_text("def greet():\n    return 'hello'\n")
    (tmp_path / "big.bin").write_bytes(b"\x00" * 10_000_000)  # should be skipped

    r = Retriever(project_path=tmp_path)
    r.build()

    hits = r.search("how does the resampler convert 48khz audio", k=2)
    assert len(hits) >= 1
    assert "audio.py" in hits[0].file
    assert "resample" in hits[0].snippet


def test_retriever_ignores_unwanted_directories(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "noise.js").write_text("console.log('resample');\n")
    (tmp_path / "code.py").write_text("def resample():\n    pass\n")

    r = Retriever(project_path=tmp_path)
    r.build()

    hits = r.search("resample", k=5)
    assert any("code.py" in h.file for h in hits)
    assert not any("node_modules" in h.file for h in hits)
```

- [ ] **Step 3: Run, verify fail**

- [ ] **Step 4: Implement `src/rctx/retriever.py`**

```python
"""BM25 retriever over a project's source files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

_INCLUDE_EXTS = {
    ".py", ".swift", ".md", ".txt", ".rst",
    ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".toml", ".yaml", ".yml", ".json",
    ".sh", ".bash",
}
_EXCLUDE_DIR_NAMES = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".build", ".swiftpm", "DerivedData", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
_MAX_FILE_BYTES = 500_000
_CHUNK_LINES = 50
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RetrievalHit:
    file: str
    line_start: int  # 1-indexed
    line_end: int    # 1-indexed, inclusive
    snippet: str
    score: float


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        out.append(tok)
        for sub in re.split(r"(?<=[a-z0-9])(?=[A-Z])|_+", m.group(0)):
            if sub and sub.lower() != tok:
                out.append(sub.lower())
    return out


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in _INCLUDE_EXTS:
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


class Retriever:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path.resolve()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[tuple[str, int, int, str]] = []

    def build(self) -> None:
        corpus, chunks = [], []
        for fp in _iter_files(self.project_path):
            try:
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            rel = str(fp.relative_to(self.project_path))
            for start in range(0, len(lines), _CHUNK_LINES):
                end = min(start + _CHUNK_LINES, len(lines))
                snippet = "\n".join(lines[start:end])
                if not snippet.strip():
                    continue
                corpus.append(_tokenize(snippet + " " + rel))
                chunks.append((rel, start + 1, end, snippet))
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        self._chunks = chunks

    def search(self, query: str, *, k: int = 5) -> list[RetrievalHit]:
        if not self._bm25 or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[RetrievalHit] = []
        for idx in idxs[:k]:
            if scores[idx] <= 0:
                continue
            rel, s, e, snippet = self._chunks[idx]
            hits.append(RetrievalHit(file=rel, line_start=s, line_end=e,
                                     snippet=snippet, score=float(scores[idx])))
        return hits
```

- [ ] **Step 5: Run, verify pass**

Run: `uv run pytest Tests/test_retriever.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rctx/retriever.py Tests/test_retriever.py pyproject.toml
git commit -m "feat(rctx): BM25 retriever over project source files"
```

---

### Task 5: answerer.py — long-lived `claude --resume` subprocess

**Files:**
- Create: `src/rctx/answerer.py`
- Create: `Tests/test_answerer.py`
- Create: `scripts/claude_stream_diag.py` (diagnostic script — committed separately)

**Purpose:** Manage a persistent `claude --resume <id> --print --input-format stream-json --output-format stream-json` subprocess. For each `QuestionEvent`, send a JSON user message on stdin and stream back assistant text deltas as `ResponseChunk` events.

**IMPORTANT:** Claude Code's stream-json schema can change between versions. **Before implementing, run a diagnostic script** to capture the exact wire format of assistant messages in this user's installed `claude` version. Adapt the parsing in `answerer.py` accordingly.

- [ ] **Step 1: Write `scripts/claude_stream_diag.py`**

```python
#!/usr/bin/env python3
"""Diagnostic: send one message to `claude --resume` in stream-json mode
and dump every line of stdout to see the exact schema."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--message", default="ping: please reply with the single word 'pong'.")
    args = ap.parse_args()

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "--resume", args.session_id,
        "--print",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--model", args.model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout and proc.stderr

    # Try two candidate input schemas — both documented in recent Claude Code versions.
    # We'll use the newer schema and log if it errors.
    user_msg = {
        "type": "user",
        "message": {"role": "user", "content": args.message},
    }
    proc.stdin.write((json.dumps(user_msg) + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()

    print("--- STDOUT (one JSON object per line) ---", flush=True)
    async for line in proc.stdout:
        decoded = line.decode(errors="replace").rstrip()
        print(decoded, flush=True)

    rc = await proc.wait()
    err = (await proc.stderr.read()).decode(errors="replace")
    if err.strip():
        print("--- STDERR ---", flush=True)
        print(err, flush=True)
    print(f"--- exit={rc} ---", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Commit the diagnostic script separately:

```bash
chmod +x scripts/claude_stream_diag.py
git add scripts/claude_stream_diag.py
git commit -m "chore(rctx): diagnostic for claude stream-json schema"
```

- [ ] **Step 2: Run the diagnostic and INSPECT THE OUTPUT**

```bash
# Get current session ID (this conversation)
SID=$(ls -t ~/.claude/projects/-Users-darwashi-Downloads-interview-realtime-context-tui/*.jsonl | head -1 | xargs -I{} sh -c 'head -1 "{}" | python3 -c "import json,sys;print(json.loads(sys.stdin.read())[\"sessionId\"])"')
echo "session: $SID"
python scripts/claude_stream_diag.py --session-id "$SID"
```

**READ THE OUTPUT.** Note:
- The exact shape of `type=assistant` / `type=content_block_delta` messages
- Where text deltas live (likely `delta.text` or `message.content[0].text`)
- Which message type signals end-of-stream (likely `type=result` or `type=message_stop`)
- Whether non-streaming assistant blobs come through instead

Update the parsing in `answerer.py` Step 4 below to match what you observed. The reference code assumes the current-as-of-writing schema; real schemas may differ.

- [ ] **Step 3: Write failing test `Tests/test_answerer.py`**

We mock the subprocess with a `MockProcess` fixture that emits canned lines, so the test is deterministic regardless of schema drift.

```python
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
    # Canned Claude Code stream-json output.
    # NOTE: schemas can drift. If your diag shows different shapes, update
    # BOTH answerer._parse_line AND these canned lines consistently.
    lines = [
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Re"}]}}).encode() + b"\n",
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "sampling uses AVAudioConverter."}]}}).encode() + b"\n",
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
```

- [ ] **Step 4: Run, verify fail (ImportError)**

- [ ] **Step 5: Implement `src/rctx/answerer.py`**

**Before writing, update the `_parse_line` function below to match the schema you observed in Step 2.** The reference code uses the most common schema; adjust field paths if your diagnostic shows different.

```python
"""Long-lived `claude --resume` subprocess wrapper: QuestionEvent → ResponseChunk."""

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

    IMPORTANT: Schema may drift. Adjust field paths to match the output of
    scripts/claude_stream_diag.py for your installed claude version.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, False

    kind = obj.get("type")

    # End-of-response signals (adjust if needed)
    if kind in ("result", "message_stop"):
        return None, True

    # Text delta shapes — try both common forms:
    # (a) {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
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

    # (b) {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
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
        self._proc = await asyncio.create_subprocess_exec(
            "claude",
            "--resume", self._session_id,
            "--print",
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
```

- [ ] **Step 6: Run, verify pass**

Run: `uv run pytest Tests/test_answerer.py -v`

If the test fails because the assistant message shape in the canned lines doesn't match what `_parse_line` expects, update BOTH (the canned lines in the test AND the field-path logic in `_parse_line`) to match what the diagnostic showed in Step 2.

- [ ] **Step 7: Commit**

```bash
git add src/rctx/answerer.py Tests/test_answerer.py
git commit -m "feat(rctx): long-lived claude-resume subprocess answerer"
```

---

### Task 6: UI — add Q+A pane

**Files:**
- Modify: `src/rctx/ui.py`
- Modify: `Tests/test_ui.py`

- [ ] **Step 1: Replace `src/rctx/ui.py`** with the four-pane layout (preserves all existing public methods):

```python
"""Textual TUI for live transcription + grounded Q+A."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, RichLog, Static

from .events import QuestionEvent, ResponseChunk, UtteranceEvent


class TranscribeApp(App):
    """Header / Transcript (2fr green) / Interim (dim) / Q+A (3fr cyan) / Status."""

    CSS = """
    Screen { layout: vertical; }
    RichLog#transcript { height: 2fr; border: solid green; padding: 0 1; }
    Static#interim { height: auto; min-height: 1; padding: 0 1; color: $text-muted; }
    RichLog#qa { height: 3fr; border: solid cyan; padding: 0 1; }
    Static#status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._transcript_lines: list[str] = []
        self._interim_text: str = ""
        self._status_text: str = "starting…"
        self._qa_lines: list[str] = []
        self._current_response_buf: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", markup=True, highlight=False, wrap=True)
        yield Static("", id="interim")
        yield RichLog(id="qa", markup=True, highlight=False, wrap=True)
        yield Static("starting…", id="status")

    # --- public API ---

    def append_event(self, event: UtteranceEvent) -> None:
        if event.is_final:
            self.query_one("#transcript", RichLog).write(event.text)
            self._transcript_lines.append(event.text)
            self._interim_text = ""
            self.query_one("#interim", Static).update("")
        else:
            self._interim_text = event.text
            self.query_one("#interim", Static).update(f"[dim italic]{event.text}[/dim italic]")

    def set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", Static).update(text)

    def on_question_detected(self, question: QuestionEvent, question_id: int) -> None:
        line = f"[bold cyan]Q{question_id}:[/bold cyan] {question.text}"
        self.query_one("#qa", RichLog).write(line)
        self._qa_lines.append(line)
        self._current_response_buf[question_id] = ""

    def on_response_chunk(self, chunk: ResponseChunk) -> None:
        qa = self.query_one("#qa", RichLog)
        if chunk.is_final:
            if chunk.citations:
                cites = ", ".join(
                    f"{c.file}:{c.line_start}-{c.line_end}" for c in chunk.citations
                )
                qa.write(f"[dim]↳ {cites}[/dim]")
                self._qa_lines.append(f"↳ {cites}")
            qa.write("")
            self._qa_lines.append("")
            self._current_response_buf.pop(chunk.question_id, None)
            return
        self._current_response_buf[chunk.question_id] = (
            self._current_response_buf.get(chunk.question_id, "") + chunk.text_delta
        )
        qa.write(chunk.text_delta)
        # mirror for tests: append to same shadow line if we were in the middle of one
        if (self._qa_lines
                and not self._qa_lines[-1].startswith("[bold cyan]")
                and not self._qa_lines[-1].startswith("↳")
                and self._qa_lines[-1] != ""):
            self._qa_lines[-1] += chunk.text_delta
        else:
            self._qa_lines.append(chunk.text_delta)

    # --- test helpers ---

    def transcript_text(self) -> str:
        return "\n".join(self._transcript_lines)

    def interim_text(self) -> str:
        return self._interim_text

    def status_text(self) -> str:
        return self._status_text

    def qa_text(self) -> str:
        return "\n".join(self._qa_lines)
```

- [ ] **Step 2: Add to `Tests/test_ui.py`**:

```python
import pytest

from rctx.events import Citation, QuestionEvent, ResponseChunk, UtteranceEvent
from rctx.ui import TranscribeApp


@pytest.mark.asyncio
async def test_qa_pane_renders_question_then_streaming_response_and_citations():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        utt = UtteranceEvent(text="how does x work", is_final=True, speech_final=False,
                             start_ms=0, end_ms=1000)
        q = QuestionEvent(text=utt.text, source_utterance=utt)
        app.on_question_detected(q, question_id=1)
        app.on_response_chunk(ResponseChunk(question_id=1, text_delta="It ", is_final=False))
        app.on_response_chunk(ResponseChunk(question_id=1, text_delta="works.", is_final=False))
        app.on_response_chunk(ResponseChunk(
            question_id=1, text_delta="", is_final=True,
            citations=(Citation(file="src/foo.py", line_start=1, line_end=10),),
        ))
        await pilot.pause()

        qa = app.qa_text()
        assert "Q1:" in qa
        assert "how does x work" in qa
        assert "It works." in qa
        assert "src/foo.py:1-10" in qa
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All prior tests still pass + new Q+A tests.

- [ ] **Step 4: Commit**

```bash
git add src/rctx/ui.py Tests/test_ui.py
git commit -m "feat(rctx): UI Q+A pane for streaming grounded responses"
```

---

### Task 7: Orchestrator + CLI wiring

**Files:**
- Modify: `src/rctx/orchestrator.py`
- Modify: `src/rctx/__main__.py`

- [ ] **Step 1: Replace `src/rctx/orchestrator.py`**

```python
"""Wire audio-tap → transcribe → (transcript + classifier → retriever → answerer → Q+A)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

from .answerer import Answerer
from .audio_tap import read_frames, spawn
from .classifier import is_question
from .events import QuestionEvent, UtteranceEvent
from .retriever import Retriever
from .transcribe import run as transcribe_run
from .ui import TranscribeApp


async def run(
    audio_tap_binary: Path,
    socket_path: str,
    project_path: Path,
    session_id: str,
) -> None:
    app = TranscribeApp()
    qa_counter = itertools.count(1)

    async def pump() -> None:
        app.set_status(f"indexing {project_path}…")
        retriever = Retriever(project_path=project_path)
        retriever.build()

        app.set_status(f"starting claude --resume {session_id[:8]}…")
        answerer = Answerer(session_id=session_id)
        await answerer.start()

        app.set_status(f"spawning audio-tap → {socket_path}")
        proc = await spawn(audio_tap_binary, socket_path)
        try:
            app.set_status(f"ready. session={session_id[:8]}…")
            frames = read_frames(socket_path)
            async for ev in transcribe_run(frames):
                app.append_event(ev)
                if ev.is_final and is_question(ev):
                    asyncio.create_task(_handle_question(ev, retriever, answerer, app, qa_counter))
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
        app.set_status(f"RAG error: {exc!r}")
```

- [ ] **Step 2: Update `src/rctx/__main__.py`** — add `--session-id` argument and auto-detect:

Replace the full file:

```python
"""rctx CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .orchestrator import run as run_orchestrator
from .session_finder import find_most_recent_session


def _find_audio_tap_binary() -> Path:
    here = Path(__file__).resolve()
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
    parser.add_argument("--project", type=Path, required=True,
                        help="Project dir (its Claude Code session will be resumed).")
    parser.add_argument("--socket-path", default="/tmp/rctx.sock")
    parser.add_argument("--audio-tap", type=Path, default=None)
    parser.add_argument("--session-id", default=None,
                        help="Override auto-detected Claude session ID.")
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("rctx: DEEPGRAM_API_KEY not set.", file=sys.stderr)
        return 2
    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    sid = args.session_id or find_most_recent_session(args.project)
    if not sid:
        print(
            f"rctx: no Claude session found for {args.project}. "
            "Start one with `claude` in that directory first, or pass --session-id.",
            file=sys.stderr,
        )
        return 2

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
                session_id=sid,
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke the CLI**

Run: `uv run rctx --help` — help printed.

Run: `uv run rctx --project /nonexistent` — exits 2 with proper error (assumes `DEEPGRAM_API_KEY` is set).

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (≥15 total).

- [ ] **Step 5: Commit**

```bash
git add src/rctx/orchestrator.py src/rctx/__main__.py
git commit -m "feat(rctx): orchestrator wires classifier + retriever + answerer"
```

---

### Task 8: End-to-end smoke test (manual)

- [ ] **Step 1: Launch**

```bash
uv run rctx --project /Users/darwashi/Downloads/interview/realtime-context-tui \
            --socket-path /tmp/rctx-e2e.sock
```

Expected status progression: `indexing …` → `starting claude --resume …` → `spawning audio-tap…` → `ready. session=<prefix>…`.

- [ ] **Step 2: Ask a question out loud (into a speaker pointed at the mic, or via system audio)**

Speak something like: *"How does the resampler handle 48 kilohertz audio?"*

Expected within a few seconds:
- Utterance appears in Transcript pane.
- `Q1: how does the resampler handle 48 kilohertz audio?` appears in the Q+A pane in cyan.
- Claude's answer streams in below (from the resumed session — should draw on the build conversation).
- A `↳ Sources/AudioTap/Resampler.swift:1-50, …` citation line appears after streaming completes.

- [ ] **Step 3: Say a non-question**

*"yeah, that makes sense"* should appear in the Transcript ONLY — not in the Q+A pane.

- [ ] **Step 4: Quit & push**

`q` to quit. Then:

```bash
git push origin main
```

---

## Done criteria for Plan 3

- `uv run pytest` passes (≥15 tests).
- `rctx` launches without any API keys other than `DEEPGRAM_API_KEY`.
- Questions trigger streaming grounded answers in the Q+A pane with file:line citations.
- Non-questions do not trigger answers.
- All commits on `main`, pushed.
