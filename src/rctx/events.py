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
