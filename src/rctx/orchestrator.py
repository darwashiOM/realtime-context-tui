"""Wire audio-tap -> transcribe -> (transcript + classifier -> retriever -> answerer -> Q+A)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

from .answerer import Answerer
from .audio_tap import read_frames, spawn
from .classifier import is_question
from .events import QuestionEvent, UtteranceEvent
from .retriever import Retriever
from .transcribe import run as transcribe_run
from .ui import TranscribeApp


async def run(
    audio_tap_binary: Path,
    socket_path: str,
    project_path: Path,
    session_id: str,
) -> None:
    app = TranscribeApp()
    qa_counter = itertools.count(1)

    async def pump() -> None:
        app.set_status(f"indexing {project_path}…")
        retriever = Retriever(project_path=project_path)
        retriever.build()

        app.set_status(f"starting claude --resume {session_id[:8]}…")
        answerer = Answerer(session_id=session_id)
        await answerer.start()

        app.set_status(f"spawning audio-tap → {socket_path}")
        proc = await spawn(audio_tap_binary, socket_path)
        try:
            app.set_status(f"ready. session={session_id[:8]}…")
            frames = read_frames(socket_path)
            async for ev in transcribe_run(frames):
                app.append_event(ev)
                if ev.is_final and is_question(ev):
                    asyncio.create_task(_handle_question(ev, retriever, answerer, app, qa_counter))
        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            await answerer.stop()

    pump_task = asyncio.create_task(pump())
    try:
        await app.run_async()
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass


async def _handle_question(
    utterance: UtteranceEvent,
    retriever: Retriever,
    answerer: Answerer,
    app: TranscribeApp,
    counter: "itertools.count[int]",
) -> None:
    try:
        q = QuestionEvent(text=utterance.text, source_utterance=utterance)
        qid = next(counter)
        app.on_question_detected(q, question_id=qid)
        hits = retriever.search(q.text, k=5)
        async for chunk in answerer.answer(q, hits, question_id=qid):
            app.on_response_chunk(chunk)
    except Exception as exc:
        app.set_status(f"RAG error: {exc!r}")
