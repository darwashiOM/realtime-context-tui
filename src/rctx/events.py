"""Typed events flowing through the rctx pipeline."""

from dataclasses import dataclass
from enum import IntEnum


class StreamTag(IntEnum):
    THEM = 0
    ME = 1


@dataclass(frozen=True, slots=True)
class Frame:
    """One audio frame from audio-tap. PCM is 16kHz mono Int16 little-endian."""

    stream_tag: StreamTag
    timestamp_ms: int
    pcm: bytes


@dataclass(frozen=True, slots=True)
class UtteranceEvent:
    """One transcription event emitted by the transcriber.

    ``is_final`` marks the last interim result for a clause (won't be amended).
    ``speech_final`` marks the end of a complete utterance (speaker stopped).
    """

    text: str
    is_final: bool
    speech_final: bool
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class QuestionEvent:
    """A finalized utterance classified as a question to the presenter."""

    text: str
    source_utterance: UtteranceEvent


@dataclass(frozen=True, slots=True)
class Citation:
    """A file:line reference shown in the UI below an answer."""

    file: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ResponseChunk:
    """A streamed piece of an answer.

    ``question_id`` ties chunks to the question they answer. ``is_final``
    marks the last chunk (carries the final ``citations`` tuple).
    """

    question_id: int
    text_delta: str
    is_final: bool
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True, slots=True)
class CoachChunk:
    """A streamed piece of a coach suggestion (next-utterance hint).

    ``coach_id`` ties chunks to the speech turn they suggest a continuation
    for. ``is_final`` marks the last chunk.
    """

    coach_id: int
    text_delta: str
    is_final: bool
