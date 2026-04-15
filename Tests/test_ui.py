import pytest

from rctx.ui import TranscribeApp
from rctx.events import Citation, QuestionEvent, ResponseChunk, UtteranceEvent


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
async def test_is_final_commits_to_transcript():
    # Diagnostic capture showed Deepgram's clause-level is_final events
    # (is_final=True, speech_final=False) arrive with clean, non-overlapping
    # start timestamps at natural clause boundaries. They should commit to
    # the transcript history; waiting for speech_final (which needs >1s of
    # silence) leaves normal continuous speech stuck in the interim pane.
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.append_event(UtteranceEvent(
            text="the key that will", is_final=True, speech_final=False,
            start_ms=0, end_ms=800,
        ))
        await pilot.pause()
        assert "the key that will" in app.transcript_text()
        assert app.interim_text() == ""


@pytest.mark.asyncio
async def test_app_set_status_updates_status_bar():
    app = TranscribeApp()
    async with app.run_test() as pilot:
        app.set_status("READY")
        await pilot.pause()
        assert app.status_text() == "READY"


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
