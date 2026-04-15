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

    def __init__(self) -> None:
        super().__init__()
        # Mirror of what we've written to the transcript, used by test helpers
        # (textual's RichLog renders via Strip internals; reading back ".lines"
        # only yields content once a render pass has happened, which is flaky
        # in Pilot-driven tests).
        self._transcript_lines: list[str] = []
        self._status_text: str = "starting…"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", markup=True, highlight=False, wrap=True)
        yield Static(self._status_text, id="status")

    # --- Public API used by orchestrator ---

    def append_event(self, event: UtteranceEvent) -> None:
        log = self.query_one("#transcript", RichLog)
        if event.is_final:
            rendered = event.text
        else:
            rendered = f"[dim]{event.text}[/dim]"
        log.write(rendered)
        self._transcript_lines.append(event.text)

    def set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", Static).update(text)

    # --- Test helpers ---

    def transcript_text(self) -> str:
        return "\n".join(self._transcript_lines)

    def status_text(self) -> str:
        return self._status_text
