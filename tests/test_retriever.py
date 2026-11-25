"""Baseline tests for backend/retriever.py: ranking, source mapping, golden retrieval, empty index."""
import json
import os
import pickle

import faiss
import numpy as np
import pytest

from backend import ingest, retriever
from tests.fakes import embed_text
from tests.pdfutil import GOLDEN_DIR, numbered_words, write_golden_pdfs, write_pdf

EXPECTED = json.loads((GOLDEN_DIR / "expected_retrieval.json").read_text())["cases"]


def _write_index(chunks: list[tuple[str, str]], vectors: np.ndarray):
    os.makedirs("data/index", exist_ok=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype("float32"))
    faiss.write_index(index, retriever.INDEX_PATH)
    with open(retriever.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def test_paths_match_ingest():
    assert (retriever.INDEX_PATH, retriever.CHUNKS_PATH) == (ingest.INDEX_PATH, ingest.CHUNKS_PATH)


class TestRankingAndMapping:
    def test_results_ordered_by_ascending_l2_distance(self, strict_external_clients):
        # 1-D geometry: query at 0.0; chunk vectors at distances 3, 1, 4, 2, 5 (stored order).
        positions = [3.0, 1.0, 4.0, 2.0, 5.0]
        vecs = np.array([[p, 0.0] for p in positions])
        chunks = [(f"doc{i}.pdf", f"text-{i}") for i in range(5)]
        _write_index(chunks, vecs)
        strict_external_clients["retriever"].models.embed_content.side_effect = None
        strict_external_clients["retriever"].models.embed_content.return_value.embeddings = [
            type("E", (), {"values": [0.0, 0.0]})()]

        out = retriever.retrieve("anything")
        # independently derived: nearest first -> stored indices 1, 3, 0, 2
        assert out == [
            {"source": "doc1.pdf", "text": "text-1"},
            {"source": "doc3.pdf", "text": "text-3"},
            {"source": "doc0.pdf", "text": "text-0"},
            {"source": "doc2.pdf", "text": "text-2"},
        ]

    def test_default_top_k_is_4_and_top_k_is_respected(self, embedder):
        chunks = [(f"d{i}.pdf", f"unique{i} token{i} " * 10) for i in range(10)]
        _write_index(chunks, np.array([embed_text(t) for _, t in chunks]))
        assert len(retriever.retrieve("unique3")) == 4
        assert len(retriever.retrieve("unique3", top_k=2)) == 2
        assert len(retriever.retrieve("unique3", top_k=10)) == 10

    def test_exact_match_maps_to_correct_chunk_and_source(self, embedder):
        chunks = [(f"d{i}.pdf", f"unique{i} token{i} " * 10) for i in range(10)]
        _write_index(chunks, np.array([embed_text(t) for _, t in chunks]))
        for i in (0, 4, 9):
            top = retriever.retrieve(chunks[i][1])[0]
            assert top == {"source": f"d{i}.pdf", "text": chunks[i][1]}

    def test_result_shape(self, embedder):
        chunks = [("a.pdf", "alpha " * 20), ("b.pdf", "beta " * 20)]
        _write_index(chunks, np.array([embed_text(t) for _, t in chunks]))
        for item in retriever.retrieve("alpha", top_k=2):
            assert set(item) == {"source", "text"}

    def test_query_is_embedded_with_gemini_embedding_model(self, embedder):
        _write_index([("a.pdf", "alpha " * 20)], np.array([embed_text("alpha " * 20)]))
        retriever.retrieve("alpha query", top_k=1)
        assert embedder.calls[-1] == {"model": "gemini-embedding-001", "contents": "alpha query"}

    def test_intra_document_chunk_selection(self, isolated_workdir, embedder):
        # 1400 numbered words -> exactly two chunks [0,800) and [700,1400);
        # w1200/w1250/w1300 occur only in the second chunk.
        words = numbered_words(1400).split()
        write_pdf(isolated_workdir / "data" / "pdfs" / "long.pdf",
                  [" ".join(words[i:i + 350]) for i in range(0, 1400, 350)])
        ingest.build_index()
        top = retriever.retrieve("w1200 w1250 w1300", top_k=1)[0]
        assert top["source"] == "long.pdf"
        assert top["text"].split()[0] == "w700"


class TestGoldenRetrieval:
    """Pipeline-wiring check against independently defined expectations (fake embedder only)."""

    @pytest.fixture
    def golden_index(self, isolated_workdir, embedder):
        write_golden_pdfs(isolated_workdir / "data" / "pdfs")
        ingest.build_index()

    @pytest.mark.parametrize("case", EXPECTED, ids=[c["query"][:40] for c in EXPECTED])
    def test_expected_top_source(self, golden_index, case):
        results = retriever.retrieve(case["query"])
        assert results[0]["source"] == case["expected_top_source"]
        assert len({r["source"] for r in results}) == 4  # 4 single-chunk docs, top_k=4


class TestEmptyAndMissingIndex:
    def test_missing_index_raises(self, embedder):
        with pytest.raises(Exception):
            retriever.retrieve("anything")

    def test_index_without_chunks_file_raises(self, embedder):
        _write_index([("a.pdf", "alpha " * 20)], np.array([embed_text("alpha")]))
        os.remove(retriever.CHUNKS_PATH)
        with pytest.raises(FileNotFoundError):
            retriever.retrieve("alpha")


class TestKnownDefects:
    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-01: FAISS returns -1 for missing neighbours; "
                       "`i < len(chunks)` admits -1 so chunks[-1] is returned as a fake hit")
    def test_fewer_chunks_than_top_k_returns_only_real_chunks(self, embedder):
        chunks = [("a.pdf", "alpha " * 20), ("b.pdf", "beta " * 20)]
        _write_index(chunks, np.array([embed_text(t) for _, t in chunks]))
        out = retriever.retrieve("alpha")  # top_k=4 > ntotal=2
        assert len(out) == 2
        assert len({(r["source"], r["text"]) for r in out}) == len(out)
