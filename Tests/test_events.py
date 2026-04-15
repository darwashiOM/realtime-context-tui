from rctx.events import Citation, Frame, QuestionEvent, ResponseChunk, StreamTag, UtteranceEvent


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


def test_coach_chunk_fields():
    from rctx.events import CoachChunk

    c = CoachChunk(coach_id=3, text_delta="Try saying ", is_final=False)
    assert c.coach_id == 3
    assert c.text_delta == "Try saying "
    assert c.is_final is False
