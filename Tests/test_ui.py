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
        await pilot.pause()
        # Interim "hello" shows in the interim pane, NOT in finalized history.
        assert "hello" in app.interim_text()
        assert "hello" not in app.transcript_text()

        app.append_event(UtteranceEvent(
            text="hello world", is_final=True, speech_final=True,
            start_ms=0, end_ms=1200,
        ))
        await pilot.pause()
        # Finalized utterance lands in the transcript history...
        assert "hello world" in app.transcript_text()
        # ...and the interim pane is cleared.
        assert app.interim_text() == ""


@pytest.mark.asyncio
async def test_is_final_without_speech_final_stays_in_interim():
    # Deepgram sometimes emits clause-level finals (is_final=True,
    # speech_final=False) whose text overlaps with the next clause's interim.
    # These must NOT be committed to the transcript history; they stay in
    # the interim pane so the next update overwrites them cleanly.
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.append_event(UtteranceEvent(
            text="the key that will", is_final=True, speech_final=False,
            start_ms=0, end_ms=800,
        ))
        await pilot.pause()
        assert "the key that will" in app.interim_text()
        assert "the key that will" not in app.transcript_text()


@pytest.mark.asyncio
async def test_app_set_status_updates_status_bar():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.set_status("READY")
        await pilot.pause()
        assert app.status_text() == "READY"
