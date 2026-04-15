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
