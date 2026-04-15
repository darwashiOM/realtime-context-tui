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
