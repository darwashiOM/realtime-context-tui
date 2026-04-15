"""BM25 retriever over a project's source files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

_INCLUDE_EXTS = {
    ".py", ".swift", ".md", ".txt", ".rst",
    ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".toml", ".yaml", ".yml", ".json",
    ".sh", ".bash",
}
_EXCLUDE_DIR_NAMES = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".build", ".swiftpm", "DerivedData", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
_MAX_FILE_BYTES = 500_000
_CHUNK_LINES = 50
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RetrievalHit:
    file: str
    line_start: int  # 1-indexed
    line_end: int    # 1-indexed, inclusive
    snippet: str
    score: float


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        out.append(tok)
        for sub in re.split(r"(?<=[a-z0-9])(?=[A-Z])|_+", m.group(0)):
            if sub and sub.lower() != tok:
                out.append(sub.lower())
    return out


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in _INCLUDE_EXTS:
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


class Retriever:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path.resolve()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[tuple[str, int, int, str]] = []

    def build(self) -> None:
        corpus, chunks = [], []
        for fp in _iter_files(self.project_path):
            try:
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            rel = str(fp.relative_to(self.project_path))
            for start in range(0, len(lines), _CHUNK_LINES):
                end = min(start + _CHUNK_LINES, len(lines))
                snippet = "\n".join(lines[start:end])
                if not snippet.strip():
                    continue
                corpus.append(_tokenize(snippet + " " + rel))
                chunks.append((rel, start + 1, end, snippet))
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        self._chunks = chunks

    def search(self, query: str, *, k: int = 5) -> list[RetrievalHit]:
        if not self._bm25 or not self._chunks:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # Secondary key: raw term-overlap count. BM25Okapi gives near-zero IDF
        # (and thus zero scores) for very small corpora where every term appears
        # in every doc; in that regime we still want the chunk with the most
        # query terms to rank first.
        query_set = set(query_tokens)
        overlaps = [
            sum(1 for tok in set(_tokenize(snippet + " " + rel)) if tok in query_set)
            for rel, _s, _e, snippet in self._chunks
        ]
        idxs = sorted(
            range(len(scores)),
            key=lambda i: (scores[i], overlaps[i]),
            reverse=True,
        )
        hits: list[RetrievalHit] = []
        for idx in idxs[:k]:
            # Only drop strictly-negative BM25 scores; zero-score chunks with
            # any term overlap are still useful signal for the answerer.
            if scores[idx] < 0 and overlaps[idx] == 0:
                continue
            if scores[idx] <= 0 and overlaps[idx] == 0:
                continue
            rel, s, e, snippet = self._chunks[idx]
            hits.append(RetrievalHit(file=rel, line_start=s, line_end=e,
                                     snippet=snippet, score=float(scores[idx])))
        return hits
