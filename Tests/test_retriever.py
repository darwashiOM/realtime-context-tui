from pathlib import Path

from rctx.retriever import Retriever


def test_retriever_indexes_code_files_and_returns_relevant_chunk(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "audio.py").write_text(
        "def resample(buf):\n"
        "    # resample 48khz float32 to 16khz int16\n"
        "    return converted_bytes\n"
    )
    (tmp_path / "src" / "unrelated.py").write_text("def greet():\n    return 'hello'\n")
    (tmp_path / "big.bin").write_bytes(b"\x00" * 10_000_000)  # should be skipped

    r = Retriever(project_path=tmp_path)
    r.build()

    hits = r.search("how does the resampler convert 48khz audio", k=2)
    assert len(hits) >= 1
    assert "audio.py" in hits[0].file
    assert "resample" in hits[0].snippet


def test_retriever_ignores_unwanted_directories(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "noise.js").write_text("console.log('resample');\n")
    (tmp_path / "code.py").write_text("def resample():\n    pass\n")

    r = Retriever(project_path=tmp_path)
    r.build()

    hits = r.search("resample", k=5)
    assert any("code.py" in h.file for h in hits)
    assert not any("node_modules" in h.file for h in hits)
