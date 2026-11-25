"""Baseline tests for backend/ingest.py: extraction, chunking, embedding, indexing, persistence."""
import pickle

import faiss
import numpy as np
import pytest

from backend import ingest
from tests.fakes import DIM, embed_text
from tests.pdfutil import (GOLDEN_DOCS, golden_text, numbered_words, write_golden_pdfs,
                           write_pdf)


# ---------------------------------------------------------------- PDF text extraction
class TestParsePdf:
    def test_extracts_all_words_of_single_page(self, tmp_path):
        text = golden_text("legal_service_agreement")
        pdf = write_pdf(tmp_path / "a.pdf", [text])
        assert ingest.parse_pdf(str(pdf)).split() == text.split()

    def test_multi_page_text_is_joined_in_page_order(self, tmp_path):
        pdf = write_pdf(tmp_path / "b.pdf", ["first page alpha", "second page beta", "third page gamma"])
        out = ingest.parse_pdf(str(pdf))
        assert out.split() == "first page alpha second page beta third page gamma".split()
        assert out.index("alpha") < out.index("beta") < out.index("gamma")

    def test_blank_pdf_yields_no_words(self, tmp_path):
        pdf = write_pdf(tmp_path / "blank.pdf", [""])
        assert ingest.parse_pdf(str(pdf)).split() == []

    def test_malformed_pdf_raises(self, tmp_path):
        bad = tmp_path / "corrupt.pdf"
        bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
        with pytest.raises(RuntimeError):  # pymupdf.FileDataError subclasses RuntimeError
            ingest.parse_pdf(str(bad))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            ingest.parse_pdf(str(tmp_path / "nope.pdf"))


# ---------------------------------------------------------------- chunking and overlap
class TestChunkText:
    def test_configured_sizes(self):
        assert (ingest.CHUNK_SIZE, ingest.CHUNK_OVERLAP) == (800, 100)

    def test_empty_and_whitespace_produce_no_chunks(self):
        assert ingest.chunk_text("") == []
        assert ingest.chunk_text("   \n\t ") == []

    def test_short_text_under_threshold_is_dropped(self):
        # 50 characters or fewer is filtered out (threshold is characters, not words).
        assert ingest.chunk_text("x" * 50) == []
        assert ingest.chunk_text("x" * 51) == ["x" * 51]

    def test_single_chunk_normalises_whitespace(self):
        text = "alpha   beta\n\ngamma\tdelta " * 5
        chunks = ingest.chunk_text(text)
        assert chunks == [" ".join(text.split())]

    def test_window_positions_and_overlap(self):
        # Independently computed expectation for 1700 words, size 800, step 700:
        # windows start at 0, 700, 1400 -> [0,800), [700,1500), [1400,1700)
        words = numbered_words(1700).split()
        chunks = ingest.chunk_text(" ".join(words))
        assert [c.split() for c in chunks] == [words[0:800], words[700:1500], words[1400:1700]]
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.split()[-100:] == nxt.split()[:100], "consecutive chunks must overlap by 100 words"

    def test_every_word_is_covered(self):
        words = numbered_words(2345).split()
        covered = set()
        for c in ingest.chunk_text(" ".join(words)):
            covered.update(c.split())
        assert covered == set(words)

    def test_exact_chunk_size_boundary(self):
        words = numbered_words(800).split()
        chunks = ingest.chunk_text(" ".join(words))
        # step 700 < 800 so a second window [700,800) is produced (see D-07)
        assert chunks[0].split() == words


# ---------------------------------------------------------------- embedding (injected mock)
class TestEmbedding:
    def test_get_embedding_calls_sdk_with_model_and_returns_values(self, embedder):
        vec = ingest.get_embedding("annual leave")
        assert embedder.calls == [{"model": "gemini-embedding-001", "contents": "annual leave"}]
        assert np.allclose(vec, embed_text("annual leave"))
        assert len(vec) == DIM

    def test_embedding_sdk_failure_propagates(self, strict_external_clients):
        strict_external_clients["ingest"].models.embed_content.side_effect = RuntimeError("quota")
        with pytest.raises(RuntimeError, match="quota"):
            ingest.get_embedding("x")


# ---------------------------------------------------------------- index build and persistence
class TestBuildIndex:
    def test_builds_and_persists_index_and_chunks(self, isolated_workdir, embedder):
        names = write_golden_pdfs(isolated_workdir / "data" / "pdfs")
        ingest.build_index()

        index = faiss.read_index(ingest.INDEX_PATH)
        with open(ingest.CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        assert isinstance(index, faiss.IndexFlatL2)
        assert index.d == DIM
        assert index.ntotal == len(chunks) == len(GOLDEN_DOCS)  # each golden doc is < 800 words
        assert sorted(src for src, _ in chunks) == sorted(names)
        for src, text in chunks:
            assert isinstance(src, str) and isinstance(text, str)
            assert text.split() == golden_text(src[:-4]).split()
        assert embedder.count == len(chunks)

    def test_index_row_i_is_embedding_of_chunk_i(self, isolated_workdir, embedder):
        write_golden_pdfs(isolated_workdir / "data" / "pdfs")
        ingest.build_index()
        index = faiss.read_index(ingest.INDEX_PATH)
        with open(ingest.CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        stored = index.reconstruct_n(0, index.ntotal)
        for i, (_, text) in enumerate(chunks):
            assert np.allclose(stored[i], embed_text(text), atol=1e-6)

    def test_multi_chunk_document_is_split_with_single_source(self, isolated_workdir, embedder):
        pages = [numbered_words(300, prefix=f"p{p}w") for p in range(5)]  # 1500 words
        write_pdf(isolated_workdir / "data" / "pdfs" / "long.pdf", pages)
        ingest.build_index()
        with open(ingest.CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        assert [src for src, _ in chunks] == ["long.pdf", "long.pdf", "long.pdf"]  # starts 0, 700, 1400

    def test_non_pdf_files_are_ignored(self, isolated_workdir, embedder):
        folder = isolated_workdir / "data" / "pdfs"
        write_pdf(folder / "doc.pdf", [golden_text("study_photosynthesis_notes")])
        (folder / "notes.txt").write_text("plain text " * 50)
        (folder / "UPPER.PDF").write_bytes((folder / "doc.pdf").read_bytes())  # suffix match is case-sensitive
        ingest.build_index()
        with open(ingest.CHUNKS_PATH, "rb") as f:
            assert {src for src, _ in pickle.load(f)} == {"doc.pdf"}

    def test_no_pdfs_writes_nothing(self, isolated_workdir, embedder, capsys):
        (isolated_workdir / "data" / "pdfs").mkdir(parents=True)
        ingest.build_index()
        assert "No PDFs found." in capsys.readouterr().out
        assert not (isolated_workdir / "data" / "index").exists()
        assert embedder.count == 0

    def test_blank_pdf_only_writes_nothing(self, isolated_workdir, embedder):
        write_pdf(isolated_workdir / "data" / "pdfs" / "blank.pdf", [""])
        ingest.build_index()
        assert not (isolated_workdir / "data" / "index").exists()

    @pytest.mark.characterization
    def test_missing_pdf_folder_raises(self, isolated_workdir, embedder):
        with pytest.raises(FileNotFoundError):
            ingest.build_index()

    def test_malformed_pdf_aborts_build(self, isolated_workdir, embedder):
        folder = isolated_workdir / "data" / "pdfs"
        write_pdf(folder / "good.pdf", [golden_text("enterprise_leave_policy")])
        (folder / "bad.pdf").write_bytes(b"garbage")
        with pytest.raises(RuntimeError):
            ingest.build_index()

    def test_rebuild_replaces_previous_index(self, isolated_workdir, embedder):
        folder = isolated_workdir / "data" / "pdfs"
        write_pdf(folder / "one.pdf", [golden_text("enterprise_leave_policy")])
        ingest.build_index()
        write_pdf(folder / "two.pdf", [golden_text("legal_service_agreement")])
        ingest.build_index()
        assert faiss.read_index(ingest.INDEX_PATH).ntotal == 2


class TestChunkingKnownDefects:
    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-07: a tail window wholly inside the previous chunk's "
                       "overlap is emitted as a redundant extra chunk (e.g. 701-800 words)")
    @pytest.mark.parametrize("n", [750, 800])
    def test_no_chunk_is_wholly_contained_in_previous(self, n):
        chunks = [c.split() for c in ingest.chunk_text(numbered_words(n))]
        for prev, nxt in zip(chunks, chunks[1:]):
            assert not set(nxt) <= set(prev)
