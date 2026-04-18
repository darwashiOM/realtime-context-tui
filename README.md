# realtime-context-tui

A macOS terminal tool that streams live transcription from your Mac's audio into a [textual](https://textual.textualize.io/) TUI while surfacing relevant context from your Claude Code conversation history and local source files via BM25 retrieval.

Useful when you're working alongside audio — following a lecture, listening to a podcast, on a pair-programming call, reviewing voice notes — and you want your recent Claude Code work to be queryable in the same flow, without switching windows.

## How it works

1. A Swift helper (`audio-tap`) captures system audio + mic on macOS via ScreenCaptureKit + AVAudioEngine, tagged per-stream.
2. The Python orchestrator streams the system-audio channel to Deepgram for transcription, runs a Haiku-4.5 classifier over each finalized utterance to detect topical / question-shaped speech, retrieves relevant local chunks via ripgrep + `rank-bm25`, and streams a Sonnet-4.6 response into a textual TUI pane.
3. Speculative pre-generation starts the LLM call on the first finalized clause, so responses usually appear within ~200 ms of the speaker finishing.

## Requirements

- macOS 13+ (ScreenCaptureKit)
- Python 3.12+ via [`uv`](https://docs.astral.sh/uv/)
- Swift 5.9+
- `DEEPGRAM_API_KEY`

