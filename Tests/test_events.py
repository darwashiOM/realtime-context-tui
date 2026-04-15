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
