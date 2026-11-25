"""Optional cross-encoder reranking in the application retriever (DOCINTEL_RERANKER=minilm)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from backend import ingest, retriever
from tests.pdfutil import golden_text, write_pdf

STEMS = ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report", "study_photosynthesis_notes"]


class FakeReranker:
    """Scores a passage by how many times it contains the word 'leave' (deterministic)."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def score(self, query, texts):
        self.calls.append(len(texts))
        if self.fail:
            raise RuntimeError("reranker down")
        return np.array([t.lower().count("leave") for t in texts], dtype=float)


@pytest.fixture
def indexed(isolated_workdir, embedder):
    for s in STEMS:
        write_pdf(isolated_workdir / "data/pdfs" / f"{s}.pdf", [golden_text(s)])
    ingest.build_index()


def dense(q, k=4):
    return retriever.retrieve(q, top_k=k)


def test_default_is_dense_and_reranker_never_loaded(indexed, monkeypatch):
    monkeypatch.delenv("DOCINTEL_RERANKER", raising=False)
    monkeypatch.setattr(retriever, "_RERANKER", None)
    out = retriever.retrieve("haemoglobin ferritin")
    monkeypatch.setenv("DOCINTEL_RERANKER", "off")
    assert out == retriever.retrieve("haemoglobin ferritin") and retriever._RERANKER is None


@pytest.mark.parametrize("var,val", [("DOCINTEL_RERANKER", "bm25"), ("DOCINTEL_RERANK_POOL", "0"),
                                     ("DOCINTEL_RERANK_POOL", "101"), ("DOCINTEL_RERANK_POOL", "x")])
def test_invalid_flags_rejected(indexed, monkeypatch, var, val):
    monkeypatch.setenv("DOCINTEL_RERANKER", "minilm")
    monkeypatch.setenv(var, val)
    monkeypatch.setattr(retriever, "_RERANKER", FakeReranker())
    with pytest.raises(ValueError):
        retriever.retrieve("q")


def test_reranker_reorders_bounded_pool_and_respects_top_k(indexed, monkeypatch):
    fake = FakeReranker()
    monkeypatch.setattr(retriever, "_RERANKER", fake)
    monkeypatch.setenv("DOCINTEL_RERANKER", "minilm")
    monkeypatch.setenv("DOCINTEL_RERANK_POOL", "3")
    out = retriever.retrieve("haemoglobin ferritin", top_k=2)
    assert fake.calls == [3] and len(out) == 2
    texts = [c["text"] for c in out]
    assert texts[0].lower().count("leave") >= texts[1].lower().count("leave")


def test_pool_smaller_index_excludes_faiss_padding(indexed, monkeypatch):
    fake = FakeReranker()
    monkeypatch.setattr(retriever, "_RERANKER", fake)
    monkeypatch.setenv("DOCINTEL_RERANKER", "minilm")
    monkeypatch.setenv("DOCINTEL_RERANK_POOL", "20")  # index has 4 chunks -> FAISS pads with -1
    out = retriever.retrieve("anything", top_k=4)
    assert fake.calls == [4] and len({c["source"] for c in out}) == 4
    assert out[0]["source"] == "enterprise_leave_policy.pdf"  # the 'leave' document wins the rerank


def test_reranker_failure_falls_back_to_dense_order(indexed, monkeypatch):
    monkeypatch.delenv("DOCINTEL_RERANKER", raising=False)
    expected = dense("haemoglobin ferritin")
    monkeypatch.setattr(retriever, "_RERANKER", FakeReranker(fail=True))
    monkeypatch.setenv("DOCINTEL_RERANKER", "minilm")
    before = retriever.STATS["rerank_failures"]
    assert retriever.retrieve("haemoglobin ferritin") == expected
    assert retriever.STATS["rerank_failures"] == before + 1


def _model_available():
    try:
        import onnxruntime  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
        from backend.reranker import MODEL
        return isinstance(try_to_load_from_cache(MODEL[0], "onnx/model.onnx", revision=MODEL[1]), str)
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="optional reranker model/runtime not installed")
def test_application_reranker_matches_evaluation_harness(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from backend.reranker import MiniLMReranker
    spec = importlib.util.spec_from_file_location("rre", Path(__file__).resolve().parent.parent / "evaluation/run_retrieval_eval.py")
    rre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rre)
    texts = [golden_text(s) for s in STEMS]
    app_scores = MiniLMReranker().score("How many days of annual leave carry forward?", texts)
    eval_scores = rre.CrossEncoder().score("How many days of annual leave carry forward?", texts)
    assert np.allclose(app_scores, eval_scores, atol=1e-5) and int(np.argmax(app_scores)) == 0


class ReversingReranker:
    """Scores = candidate position, so a correct implementation returns the dense pool reversed."""
    def score(self, query, texts):
        return np.arange(len(texts), dtype=float)


def test_reranked_order_follows_scores_exactly(indexed, monkeypatch):
    monkeypatch.delenv("DOCINTEL_RERANKER", raising=False)
    dense_pool = retriever.retrieve("termination notice agreement", top_k=4)  # all 4 chunks, dense order
    monkeypatch.setattr(retriever, "_RERANKER", ReversingReranker())
    monkeypatch.setenv("DOCINTEL_RERANKER", "minilm")
    monkeypatch.setenv("DOCINTEL_RERANK_POOL", "4")
    assert retriever.retrieve("termination notice agreement", top_k=4) == dense_pool[::-1]
    assert retriever.retrieve("termination notice agreement", top_k=2) == dense_pool[::-1][:2]
