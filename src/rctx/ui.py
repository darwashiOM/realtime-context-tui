"""Textual TUI for live transcription."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, RichLog, Static

from .events import UtteranceEvent


class TranscribeApp(App):
    """Four-region live-transcription view: header + finalized transcript +
    in-progress interim pane + status bar.

    The interim pane exists so Deepgram's stream of partial hypotheses
    ("this is no" -> "this is no hallucination" -> ...) updates in place
    instead of stacking as separate lines in the finalized history.
    """

    CSS = """
    Screen { layout: vertical; }
    RichLog#transcript { height: 1fr; border: solid green; padding: 0 1; }
    Static#interim { height: auto; min-height: 1; padding: 0 1; color: $text-muted; }
    Static#status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        # Shadow state mirroring what's visible, used by test helpers.
        # (Textual's RichLog renders via Strip internals; reading back ".lines"
        # only yields content once a render pass has happened, which is flaky
        # in Pilot-driven tests.)
        self._transcript_lines: list[str] = []
        self._interim_text: str = ""
        self._status_text: str = "starting…"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", markup=True, highlight=False, wrap=True)
        yield Static("", id="interim")
        yield Static(self._status_text, id="status")

    # --- Public API used by orchestrator ---

    def append_event(self, event: UtteranceEvent) -> None:
        # Only commit to the finalized transcript when speech_final is True
        # (speaker actually paused / utterance complete). Deepgram also emits
        # clause-level finals (is_final=True, speech_final=False) whose text
        # frequently OVERLAPS with the next clause's interim -- committing
        # those produces duplicated words in the history
        # (e.g. "the key the key that will put ..."). Treating them as still-
        # interim means they're overwritten in place by subsequent updates
        # until the real end-of-utterance arrives.
        if event.speech_final:
            log = self.query_one("#transcript", RichLog)
            log.write(event.text)
            self._transcript_lines.append(event.text)
            self._interim_text = ""
            self.query_one("#interim", Static).update("")
        else:
            self._interim_text = event.text
            self.query_one("#interim", Static).update(
                f"[dim italic]{event.text}[/dim italic]"
            )

    def set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", Static).update(text)

    # --- Test helpers ---

    def transcript_text(self) -> str:
        return "\n".join(self._transcript_lines)

    def interim_text(self) -> str:
        return self._interim_text

    def status_text(self) -> str:
        return self._status_text
