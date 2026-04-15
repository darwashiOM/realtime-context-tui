"""Local regex-based question classifier. Generous-triggering by design."""

from __future__ import annotations

import re

from .events import UtteranceEvent

# Start-of-sentence interrogatives (case-insensitive, must be the first word
# or follow ". "/"? "). Keep this list broad; false positives are cheap.
_INTERROGATIVE_STARTS = re.compile(
    r"(?:^|[.?!]\s+)(how|why|what|when|where|which|who|"
    r"can|could|would|should|will|do|does|did|is|are|am|was|were|"
    r"may|might|have|has|had)\b",
    re.IGNORECASE,
)


def is_question(utterance: UtteranceEvent) -> bool:
    """Return True if the utterance looks like a question to the presenter."""
    if not utterance.is_final:
        return False
    text = utterance.text.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    if _INTERROGATIVE_STARTS.search(text):
        return True
    return False
