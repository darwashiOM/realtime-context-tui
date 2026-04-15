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

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("up", "prev_question", "prev Q"),
        ("down", "next_question", "next Q"),
        ("home", "first_question", "first Q"),
        ("end", "last_question", "last Q"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._transcript_lines: list[str] = []
        self._interim_text: str = ""
        self._status_text: str = "starting…"
        self._qa_lines: list[str] = []
        self._current_response_buf: dict[int, str] = {}
        self._question_y_positions: list[int] = []

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
        qa = self.query_one("#qa", RichLog)
        # NOTE: textual 8.2.3 RichLog has no `line_count`; the rendered lines
        # are exposed as `qa.lines` (a list). Use its length as the Y offset.
        self._question_y_positions.append(len(qa.lines))
        line = f"[bold cyan]Q{question_id}:[/bold cyan] {question.text}"
        qa.write(line)
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

    # --- navigation actions ---

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

    # --- test helpers ---

    def question_y_positions(self) -> list[int]:
        return list(self._question_y_positions)

    def transcript_text(self) -> str:
        return "\n".join(self._transcript_lines)

    def interim_text(self) -> str:
        return self._interim_text

    def status_text(self) -> str:
        return self._status_text

    def qa_text(self) -> str:
        return "\n".join(self._qa_lines)
