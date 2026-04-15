# realtime-context-tui — Design

**Status:** Draft (brainstorm complete, pending implementation plan)
**Date:** 2026-04-15

## Problem

When working alongside audio — listening to a podcast, following a recorded lecture, taking voice notes, or on a pair-programming call — breaking flow to search your Claude Code conversation history or project source for context is expensive. You want relevant snippets surfaced automatically in a glanceable TUI pane while the audio plays, without leaving your terminal.

**Goal:** A macOS terminal tool that listens to audio on your Mac, detects topical / question-shaped utterances, and within ~1–2 seconds displays a streaming context summary grounded in your Claude Code conversation history and the source code of a project you point it at.

### Success criteria

- Utterance-end to first-token latency ≤ 2 s (warm cache)
- Responses cite specific `file:line` when referencing code
- Works regardless of which app is producing audio (system-level capture)
- No wake word or manual trigger — operates autonomously
- Runs on the user's Mac (except cloud transcription / LLM APIs)

### Non-goals

- Speaker diarization beyond "them" vs "me"
- Processing screen-share content or slide OCR
- Persistent storage of audio or transcripts
- Cross-platform support (macOS only)
- Recording compliance or consent enforcement — left to the user

---

## Decisions

| # | Concern | Decision | Rationale |
|---|---|---|---|
| 1 | Context source | `.jsonl` transcripts under `~/.claude/projects/<slug>/` for the project passed via `--project` | Direct file access, no scraping; scoped to a single project to reduce noise |
| 2 | Audio capture | macOS ScreenCaptureKit (system audio) + AVAudioEngine (mic), tagged per-stream | Native, no install, uniform across every app that emits audio |
| 3 | Transcription | Deepgram Nova-3 streaming primary; local `whisper.cpp` fallback | Deepgram `UtteranceEnd` is purpose-built for "speaker stopped"; strong accuracy on technical jargon |
| 4 | Trigger | Haiku 4.5 classifier per finalized utterance + speculative pre-generation | Avoids generating on small talk; speculation hides model latency |
| 5 | Retrieval | Prompt-cached full `.jsonl` dump + live ripgrep / BM25 over project source | Cache makes repeat calls cheap; BM25 provides fresh `file:line` citations |
| 6 | UI | Standalone `textual` TUI in its own terminal — panes: Transcript / Detected / Response | Keeps primary shell clean; easy to glance at |
| 7 | Response model | Sonnet 4.6, extended thinking off, `max_tokens=400`, citation-first system prompt | First-token latency matters more than reasoning depth; short responses fit the pane |
| 8 | Privacy | No guardrails on capture — responsibility is on the user | Explicit user decision; out of scope for this tool |

---

## Architecture

Two processes communicating over a Unix domain socket:

```
┌─────────────────────────────────────────────────────────────┐
│  audio-tap  (Swift binary)                                  │
│  - ScreenCaptureKit  →  system audio stream (them)          │
│  - AVAudioEngine     →  mic stream          (me)            │
│  - Frames 20 ms PCM packets, tagged {them|me}               │
│  - Writes to $XDG_RUNTIME_DIR/rctx.sock                     │
└─────────────────────────────────────────────────────────────┘
                          │ Unix socket (tagged PCM frames)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  rctx  (Python orchestrator + textual TUI)                  │
│                                                             │
│   transcribe.py   → Deepgram websocket (them stream only)   │
│       ↓                                                     │
│   classifier.py   → Haiku 4.5: is_topical?                  │
│       ↓ (if yes)                                            │
│   retriever.py    → rg + rank-bm25 over project source      │
│       ↓                                                     │
│   answerer.py     → Sonnet 4.6 streaming (speculative)      │
│       ↓                                                     │
│   ui.py (textual) → Transcript / Detected / Response panes  │
└─────────────────────────────────────────────────────────────┘
```

**Why two processes:** Swift is mandatory for ScreenCaptureKit (no mature Python bindings). Python owns iteration-heavy orchestration where Deepgram SDK, Anthropic SDK, `rank-bm25`, and `textual` are all first-class. The socket boundary is a stable interface; either half can be rewritten independently.

**Why Python (over Rust / Go):** Latency is dominated by network (Deepgram, Anthropic) and model inference, not language overhead. The TUI layer (`textual`) and RAG layer are significantly more mature in Python.

### Module responsibilities

Each module has one purpose, communicates via typed events over `asyncio` queues, and is testable in isolation.

- **`audio-tap` (Swift)** — produces tagged PCM frames. Input: none. Output over socket: `{stream: them|me, pcm: bytes, timestamp_ms: int}`.
- **`transcribe.py`** — consumes PCM frames for the `them` stream only. Output: `UtteranceEvent {text, is_final, speech_final, start_ms, end_ms}`.
- **`classifier.py`** — consumes `UtteranceEvent` where `is_final=true`. Output: `TopicalEvent {text, urgency: low|high}` or nothing.
- **`retriever.py`** — consumes `TopicalEvent`. Output: `RetrievalResult {chunks: [{file, line_range, snippet, score}]}`.
- **`answerer.py`** — consumes `TopicalEvent` + `RetrievalResult`. Streams `ResponseChunk {text, is_final, citations: [file:line]}`. Supports mid-stream cancel.
- **`ui.py`** — consumes all event types for display. No business logic; pure view layer.
- **`orchestrator.py`** — event loop wiring queues, handling speculation (fire on `is_final`, cancel-and-refire on `speech_final` when semantic diff exceeds threshold), error recovery.

---

## Data flow & latency budget

```
T=0 ms       Audio starts
             │  audio-tap → Deepgram interims every ~150 ms
             │
T≈1500 ms    Deepgram is_final for first clause
             ├─ classifier (~200 ms) → topical=true
             ├─ retriever (~50 ms)
             └─ answerer kicks off SPECULATIVELY
             │
T≈2200 ms    Sonnet first token (buffered, not yet shown)
             │
T≈3000 ms    speech_final fires (speaker stopped)
             │
             ├─ Speculation matched final utterance → reveal buffered response
             │  Perceived latency: ~0–200 ms
             │
             └─ Final utterance changed meaningfully → cancel, re-fire
                Perceived latency: ~1500–2500 ms
```

**Cold cache (first response of the session):** ~2–3 s perceived.
**Warm cache (every subsequent response):** ~0–1.2 s when speculation hits; ~1.5–2.5 s when it misses.

### Cost budget

1-hour session, ~30 triggered responses, ~80 k-token cached transcript:

| Component | Cost |
|---|---|
| Deepgram Nova-3 streaming (1 hr @ $0.0043/min) | ~$0.24 |
| Anthropic Sonnet 4.6 (mostly cache reads + ~400 output tokens × 30) | ~$0.40 |
| Haiku 4.5 classifier (~150 utterances × ~100 input tokens) | ~$0.05 |
| **Total** | **≈$0.70 / hour** |

---

## Error handling

Designed so silent failure never happens. Every failure mode produces a visible TUI banner.

| Failure | Detection | Response |
|---|---|---|
| Deepgram websocket drop | 5 s heartbeat timeout | Exponential-backoff reconnect (3 attempts); on exhaustion flip to `whisper.cpp` fallback. Banner: `⚠ cloud transcription failed, using local whisper` |
| Missing API keys at launch | Startup env var check | Refuse to launch; print which key is missing |
| Anthropic 529 overloaded | HTTP status | One retry @ 500 ms backoff; banner on repeated failure |
| Sonnet first-token > 5 s | Timer | Show "thinking…" indicator; `esc` hotkey cancels and regenerates |
| `audio-tap` crash | Broken socket / EOF | Python detects, respawns Swift binary, resubscribes. Banner: `⚠ audio-tap died — restarting` |
| ScreenCaptureKit permission denied | First-frame error | Clear instruction in TUI: "Grant Screen Recording permission in System Settings → Privacy & Security → Screen Recording" — then exit cleanly |
| Empty `.jsonl` directory | Startup scan | Warn, continue with code-only retrieval |
| Transcripts exceed cache budget (> 180 k tokens) | Token count at launch | Use most-recent N sessions that fit. Banner: `⚠ transcript truncated: using last X sessions (Y older skipped)` |
| Classifier false negative | Not programmatically detectable | Bias classifier prompt toward "treat ambiguous as topical"; hotkey `r` forces response on last utterance |
| Classifier false positive | Not programmatically detectable | Tolerable — response renders in a pane nobody's waiting on |
| Speculation guesses wrong | Semantic diff on `speech_final` | Cancel in-flight stream (Anthropic SDK abort), fire fresh call |

---

## Testing

Pragmatic, not exhaustive. Bar: "reliable enough for live use."

| Layer | Approach |
|---|---|
| `audio-tap` (Swift) | Manual smoke test with a virtual audio device + known `.wav` |
| `transcribe.py` | Golden-file test against a pre-recorded audio snippet |
| `classifier.py` | **Most critical.** ~40 labeled real utterances; assert precision ≥ 0.9 on positives; run on every prompt tweak |
| `retriever.py` | Unit tests with fixture project + canned queries |
| `answerer.py` | Snapshot tests with mocked Anthropic streaming responses; one live smoke test gated behind `RUN_LIVE=1` |
| `ui.py` | Textual `Pilot` for rendering snapshots |
| End-to-end | Manual rehearsal: play 5 canned utterances, confirm sensible responses |

---

## Packaging

```
realtime-context-tui/
├── Package.swift                  # Swift audio-tap
├── Sources/AudioTap/
├── pyproject.toml                 # Python, uv-managed
├── src/rctx/
│   ├── __main__.py                # CLI: parses --project
│   ├── transcribe.py
│   ├── classifier.py
│   ├── retriever.py
│   ├── answerer.py
│   ├── ui.py
│   └── orchestrator.py
├── tests/
├── Makefile                       # build / run / test
└── README.md
```

**Install:** `make install` builds the Swift binary, installs Python via `uv`, symlinks `~/.local/bin/rctx`.

**Run:** `rctx --project ~/code/my-project`.

**Env vars required:** `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`.

**macOS entitlements:** `com.apple.security.device.audio-input` + Screen Recording permission (granted once via System Settings, survives reboots).

**No code signing, no CI, no release pipeline.** Personal tool, single-machine install.

---

## Open questions deferred to implementation plan

- Exact wire format for the socket protocol (length-prefixed binary vs. newline-delimited JSON with base64 PCM)
- Haiku classifier prompt + evaluation-set collection strategy
- Semantic-diff threshold for speculative-cancellation decisions
- Local `whisper.cpp` model selection (`small.en` vs `distil-medium.en`) — latency/accuracy tradeoff
- Structure of the prompt-cached system block for maximum cache-hit rate across sessions
