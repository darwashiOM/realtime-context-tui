# Session Picker + Custom Instruction + Q Navigation (Plan 4)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute. Steps use checkbox (`- [ ]`) tracking.

**Goal:** Three quality-of-life upgrades to `rctx`:
1. **Interactive session picker** — at launch, list available Claude Code sessions for the project (with timestamp + first-user preview + turn count) and let the user pick.
2. **Custom instruction preamble** — after picking, prompt for optional freeform instructions that get attached to every answer request (e.g. *"be extra concise"*, *"treat me as a junior ML engineer"*).
3. **Question navigation** — in the TUI, `up`/`down` arrow keys jump between prior Q&A entries in the Q+A pane; `home`/`end` go to first/last.

**Working directory:** `/Users/darwashi/Downloads/interview/realtime-context-tui` (on `main`).

---

### Task 1: session_picker.py — list + prompt

**Files:**
- Create: `src/rctx/session_picker.py`
- Create: `Tests/test_session_picker.py`

**Purpose:** `list_sessions(project_path)` returns `SessionInfo` entries (newest first). `prompt_for_session(sessions)` prints a numbered menu on stderr and reads a choice from stdin (blocks waiting for user). Returns the chosen session_id or `None` on cancel/EOF.

- [ ] **Step 1: Write failing test `Tests/test_session_picker.py`**

```python
import io
import json
import time
from pathlib import Path

import pytest

from rctx.session_picker import (
    SessionInfo,
    list_sessions,
    prompt_for_session,
)
from rctx.session_finder import project_slug


def _write_session(sessions_dir: Path, sid: str, first_user: str, turns: int) -> Path:
    f = sessions_dir / f"{sid}.jsonl"
    lines = [json.dumps({"sessionId": sid, "type": "user",
                         "message": {"content": first_user}})]
    for i in range(turns - 1):
        lines.append(json.dumps({"sessionId": sid, "type": "assistant",
                                 "message": {"content": [{"type": "text", "text": f"a{i}"}]}}))
    f.write_text("\n".join(lines))
    return f


def test_list_sessions_sorted_newest_first(tmp_path, monkeypatch):
    proj = tmp_path / "me" / "proj"
    proj.mkdir(parents=True)
    sdir = tmp_path / ".claude" / "projects" / project_slug(proj)
    sdir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    _write_session(sdir, "old", "older question", turns=3)
    time.sleep(0.05)
    _write_session(sdir, "new", "newer question really long text that should truncate", turns=5)

    out = list_sessions(proj)
    assert [s.session_id for s in out] == ["new", "old"]
    assert out[0].turn_count == 5
    assert "newer question" in out[0].preview
    assert len(out[0].preview) <= 80


def test_list_sessions_empty_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert list_sessions(tmp_path / "nonexistent") == []


def test_prompt_for_session_returns_chosen_id(monkeypatch, capsys):
    sessions = [
        SessionInfo(session_id="s1", mtime=1.0, preview="p1", turn_count=3),
        SessionInfo(session_id="s2", mtime=2.0, preview="p2", turn_count=6),
    ]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    chosen = prompt_for_session(sessions)
    assert chosen == "s2"


def test_prompt_for_session_default_on_empty(monkeypatch):
    sessions = [SessionInfo(session_id="a", mtime=0.0, preview="p", turn_count=1)]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert prompt_for_session(sessions) == "a"


def test_prompt_for_session_cancel_returns_none(monkeypatch):
    sessions = [SessionInfo(session_id="a", mtime=0.0, preview="p", turn_count=1)]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "0")
    assert prompt_for_session(sessions) is None
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `src/rctx/session_picker.py`**

```python
"""Interactive terminal picker for which Claude session to resume."""

from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .session_finder import project_slug


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    mtime: float
    preview: str
    turn_count: int


def list_sessions(project_path: Path) -> list[SessionInfo]:
    """Return Claude sessions for ``project_path`` newest first."""
    home = Path(os.environ.get("HOME", str(Path.home())))
    sessions_dir = home / ".claude" / "projects" / project_slug(project_path)
    if not sessions_dir.exists():
        return []

    out: list[SessionInfo] = []
    for sf in sessions_dir.glob("*.jsonl"):
        try:
            mtime = sf.stat().st_mtime
            lines = sf.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        sid: str | None = None
        first_user_text: str | None = None
        turn_count = 0
        for raw in lines:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if sid is None:
                sid = entry.get("sessionId")
            if entry.get("type") in ("user", "assistant"):
                turn_count += 1
                if first_user_text is None and entry.get("type") == "user":
                    content = entry.get("message", {}).get("content", "")
                    if isinstance(content, str):
                        first_user_text = content
                    elif isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                first_user_text = blk.get("text", "")
                                break
        if not sid:
            continue
        preview = (first_user_text or "(no user messages)").replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        out.append(SessionInfo(session_id=sid, mtime=mtime,
                               preview=preview, turn_count=turn_count))

    out.sort(key=lambda s: s.mtime, reverse=True)
    return out


def prompt_for_session(sessions: list[SessionInfo]) -> str | None:
    """Numbered menu on stderr; read stdin. Return chosen session_id or None."""
    if not sessions:
        return None
    print("", file=sys.stderr)
    print("Available Claude Code sessions:", file=sys.stderr)
    for i, s in enumerate(sessions, start=1):
        ts = datetime.datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {ts}  turns={s.turn_count:3d}  {s.preview}", file=sys.stderr)
    print("  [0] cancel", file=sys.stderr)

    while True:
        try:
            choice = input(f"Pick [1-{len(sessions)}] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice == "":
            return sessions[0].session_id
        if choice == "0":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx].session_id
        except ValueError:
            pass
        print(f"Invalid. 1-{len(sessions)}, or 0 to cancel.", file=sys.stderr)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest Tests/test_session_picker.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/session_picker.py Tests/test_session_picker.py
git commit -m "feat(rctx): interactive session picker"
```

---

### Task 2: custom instruction plumbing

**Files:**
- Modify: `src/rctx/answerer.py`
- Modify: `Tests/test_answerer.py`

**Purpose:** `Answerer` accepts a `custom_instruction: str = ""` kwarg. When non-empty, it's prepended to every `_build_user_turn` payload.

- [ ] **Step 1: Update `_build_user_turn` signature and body in `src/rctx/answerer.py`**

Replace the existing `_build_user_turn` function with:

```python
def _build_user_turn(
    q: QuestionEvent,
    hits: Sequence[RetrievalHit],
    *,
    custom_instruction: str = "",
) -> str:
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
        "- `path/file.ext:line_start-line_end` — <3-7 word description>",
        "- `path/file.ext:line_start-line_end` — <3-7 word description>",
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
```

- [ ] **Step 2: Thread `custom_instruction` through `Answerer`**

In `Answerer.__init__`, add `custom_instruction: str = ""` kwarg and store on `self._custom_instruction`. In `Answerer.answer`, pass it to `_build_user_turn(..., custom_instruction=self._custom_instruction)`.

- [ ] **Step 3: Add a test to `Tests/test_answerer.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest Tests/test_answerer.py -v`
Expected: All prior tests pass + new test passes.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/answerer.py Tests/test_answerer.py
git commit -m "feat(rctx): custom session-level instruction threaded through answerer"
```

---

### Task 3: UI — question navigation hotkeys

**Files:**
- Modify: `src/rctx/ui.py`
- Modify: `Tests/test_ui.py`

**Purpose:** Track Y-line position of each `Q#:` marker in the Q+A RichLog. Bind `up`/`down` to jump to prev/next question, `home`/`end` to first/last.

- [ ] **Step 1: Update `src/rctx/ui.py`**

Add to `__init__`:
```python
self._question_y_positions: list[int] = []
```

Update `BINDINGS`:
```python
BINDINGS = [
    ("q", "quit", "Quit"),
    ("up", "prev_question", "prev Q"),
    ("down", "next_question", "next Q"),
    ("home", "first_question", "first Q"),
    ("end", "last_question", "last Q"),
]
```

Update `on_question_detected` to capture the line position BEFORE writing the header, then record it:
```python
def on_question_detected(self, question: QuestionEvent, question_id: int) -> None:
    qa = self.query_one("#qa", RichLog)
    self._question_y_positions.append(qa.line_count)
    line = f"[bold cyan]Q{question_id}:[/bold cyan] {question.text}"
    qa.write(line)
    self._qa_lines.append(line)
    self._current_response_buf[question_id] = ""
```

Add action handlers:
```python
def _scroll_qa_to(self, y: int) -> None:
    qa = self.query_one("#qa", RichLog)
    qa.scroll_to(y=y, animate=False)

def action_prev_question(self) -> None:
    qa = self.query_one("#qa", RichLog)
    current_y = qa.scroll_y
    prev = [y for y in self._question_y_positions if y < current_y]
    if prev:
        self._scroll_qa_to(prev[-1])

def action_next_question(self) -> None:
    qa = self.query_one("#qa", RichLog)
    current_y = qa.scroll_y
    nxt = [y for y in self._question_y_positions if y > current_y]
    if nxt:
        self._scroll_qa_to(nxt[0])

def action_first_question(self) -> None:
    if self._question_y_positions:
        self._scroll_qa_to(self._question_y_positions[0])

def action_last_question(self) -> None:
    if self._question_y_positions:
        self._scroll_qa_to(self._question_y_positions[-1])
```

Also add a test helper:
```python
def question_y_positions(self) -> list[int]:
    return list(self._question_y_positions)
```

- [ ] **Step 2: Add test `Tests/test_ui.py`**

```python
@pytest.mark.asyncio
async def test_question_y_positions_track_each_question():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        utt = UtteranceEvent(text="q1", is_final=True, speech_final=False,
                             start_ms=0, end_ms=500)
        app.on_question_detected(QuestionEvent(text="q1", source_utterance=utt), 1)
        app.on_response_chunk(ResponseChunk(question_id=1, text_delta="ans", is_final=False))
        app.on_response_chunk(ResponseChunk(question_id=1, text_delta="", is_final=True))

        app.on_question_detected(QuestionEvent(text="q2", source_utterance=utt), 2)
        await pilot.pause()

        ys = app.question_y_positions()
        assert len(ys) == 2
        assert ys[0] < ys[1]  # Q2 appears after Q1 in the log
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest Tests/test_ui.py -v`
Expected: All prior UI tests + new one pass.

- [ ] **Step 4: Commit**

```bash
git add src/rctx/ui.py Tests/test_ui.py
git commit -m "feat(rctx): up/down/home/end jump between Q&A entries"
```

---

### Task 4: CLI — picker + instruction prompt wired into `__main__`

**Files:**
- Modify: `src/rctx/__main__.py`
- Modify: `src/rctx/orchestrator.py`

- [ ] **Step 1: Replace `src/rctx/__main__.py`**

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
from .session_picker import list_sessions, prompt_for_session


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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--socket-path", default="/tmp/rctx.sock")
    parser.add_argument("--audio-tap", type=Path, default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--custom-instruction", default=None,
                        help="Preamble for every answer. If omitted, prompt for one.")
    parser.add_argument("--no-picker", action="store_true",
                        help="Skip interactive picker; use most-recent session.")
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("rctx: DEEPGRAM_API_KEY not set.", file=sys.stderr)
        return 2
    if not args.project.exists():
        print(f"rctx: --project path does not exist: {args.project}", file=sys.stderr)
        return 2

    # Session selection
    sid = args.session_id
    if sid is None:
        if args.no_picker:
            sid = find_most_recent_session(args.project)
        else:
            sessions = list_sessions(args.project)
            if not sessions:
                print(f"rctx: no Claude sessions for {args.project}. "
                      "Start one with `claude` in that directory first.", file=sys.stderr)
                return 2
            sid = prompt_for_session(sessions)
    if not sid:
        print("rctx: no session chosen, exiting.", file=sys.stderr)
        return 2

    # Custom instruction prompt
    instruction = args.custom_instruction
    if instruction is None:
        print("\nOptional session-level instruction for Claude "
              "(empty to skip, e.g. 'be extra concise'):", file=sys.stderr)
        try:
            instruction = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            instruction = ""

    binary = args.audio_tap or _find_audio_tap_binary()

    try:
        asyncio.run(
            run_orchestrator(
                audio_tap_binary=binary,
                socket_path=args.socket_path,
                project_path=args.project,
                session_id=sid,
                custom_instruction=instruction or "",
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update `src/rctx/orchestrator.py` `run` signature**

Change the signature to accept `custom_instruction`, and pass it to `Answerer`:

```python
async def run(
    audio_tap_binary: Path,
    socket_path: str,
    project_path: Path,
    session_id: str,
    custom_instruction: str = "",
) -> None:
    ...
    answerer = Answerer(session_id=session_id, custom_instruction=custom_instruction)
    ...
```

- [ ] **Step 3: Smoke CLI**

Run: `uv run rctx --help`
Expected: new `--session-id`, `--custom-instruction`, `--no-picker` flags appear.

Run: `env -u DEEPGRAM_API_KEY uv run rctx --project /tmp` → exits 2.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rctx/__main__.py src/rctx/orchestrator.py
git commit -m "feat(rctx): CLI integrates picker + custom instruction prompt"
```

---

### Task 5: End-to-end smoke test (manual)

- [ ] **Step 1: Launch**

```bash
uv run rctx --project /Users/darwashi/Downloads/interview/realtime-context-tui
```

Expected in terminal (BEFORE the TUI starts):
- A numbered list of available sessions with timestamps and first-user-message previews
- Prompt: `Pick [1-N] (default 1):` — hit a number + Enter
- Prompt: `Optional session-level instruction for Claude` — type something like `"be extra concise, never use a word longer than two syllables"` + Enter, or just Enter to skip

Then the TUI launches as normal.

- [ ] **Step 2: Ask a question**

Expected: answer appears in the Q+A pane. If you gave a custom instruction (e.g. "be extra concise"), confirm it's reflected in the answer style.

- [ ] **Step 3: Ask 3+ more questions**

Each shows up with `Q1:`, `Q2:`, etc.

- [ ] **Step 4: Test navigation**

Press `up` — scroll should jump to previous Q. Press `up` again — further back. Press `down` — forward. Press `home` — jump to first Q. Press `end` — latest.

- [ ] **Step 5: Quit & push**

`q` then:

```bash
git push origin main
```

---

## Done criteria for Plan 4

- Launching `rctx` without `--session-id` shows a picker.
- Custom instruction prompt accepts freeform text and is reflected in answers.
- `up`/`down`/`home`/`end` navigate the Q+A pane.
- All tests green.
- Committed + pushed to `main`.
