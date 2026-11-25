"""Baseline API tests for backend/main.py (FastAPI) -- upload, chat, health, CORS, errors.

All calls run in-process via TestClient; embeddings and LLMs are injected fakes.
"""
import os
from unittest import mock

import faiss
import pytest
from fastapi.testclient import TestClient

from backend import ingest, llm, main
from tests.pdfutil import GOLDEN_DOCS, golden_text, make_pdf_bytes

NO_DOCS = {"answer": "No documents indexed yet. Please upload a PDF first.",
           "model": "", "confidence": "low", "sources": []}


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def upload(client, name, data=None, content_type="application/pdf"):
    if data is None:
        data = make_pdf_bytes([golden_text(name[:-4])])
    return client.post("/upload", files={"file": (name, data, content_type)})


@pytest.fixture
def golden_uploaded(client, embedder):
    for stem in GOLDEN_DOCS:
        assert upload(client, f"{stem}.pdf").status_code == 200
    return client


# ---------------------------------------------------------------- health / CORS
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_route_inventory():
    routes = {(m, r.path) for r in main.app.routes if hasattr(r, "methods") for m in r.methods}
    assert {("GET", "/health"), ("POST", "/upload"), ("POST", "/chat")} <= routes


def test_cors_allows_vite_dev_origin_only(client):
    ok = client.options("/chat", headers={"Origin": "http://localhost:5173",
                                          "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/chat", headers={"Origin": "http://evil.example",
                                           "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


# ---------------------------------------------------------------- upload
class TestUpload:
    def test_upload_saves_and_indexes(self, client, embedder, isolated_workdir):
        r = upload(client, "enterprise_leave_policy.pdf")
        assert r.status_code == 200
        assert r.json() == {"message": "enterprise_leave_policy.pdf uploaded and indexed successfully"}
        assert (isolated_workdir / "data" / "pdfs" / "enterprise_leave_policy.pdf").is_file()
        assert faiss.read_index(ingest.INDEX_PATH).ntotal == 1

    def test_upload_accumulates_documents(self, golden_uploaded):
        assert faiss.read_index(ingest.INDEX_PATH).ntotal == len(GOLDEN_DOCS)

    def test_same_filename_overwrites(self, client, embedder, isolated_workdir):
        upload(client, "doc.pdf", make_pdf_bytes([golden_text("legal_service_agreement")]))
        upload(client, "doc.pdf", make_pdf_bytes([golden_text("study_photosynthesis_notes")]))
        assert os.listdir(isolated_workdir / "data" / "pdfs") == ["doc.pdf"]
        assert faiss.read_index(ingest.INDEX_PATH).ntotal == 1

    def test_missing_file_field_is_422(self, client):
        assert client.post("/upload").status_code == 422

    def test_malformed_pdf_is_not_reported_as_success(self, client, embedder, isolated_workdir):
        # Baseline returns HTTP 500 (unhandled exception); any non-2xx satisfies the contract.
        r = upload(client, "corrupt.pdf", b"not a pdf")
        assert not 200 <= r.status_code < 300
        assert not (isolated_workdir / "data" / "index").exists()


# ---------------------------------------------------------------- chat
class TestChat:
    def test_empty_index_returns_guidance_message(self, client, strict_external_clients):
        r = client.post("/chat", json={"question": "anything?"})
        assert r.status_code == 200 and r.json() == NO_DOCS
        strict_external_clients["gemini"].models.generate_content.assert_not_called()

    def test_validation(self, client):
        assert client.post("/chat", json={}).status_code == 422
        assert client.post("/chat", json={"mode": "legal"}).status_code == 422
        assert client.post("/chat", content="not json",
                           headers={"content-type": "application/json"}).status_code == 422

    def test_response_contract(self, golden_uploaded, gemini_ok):
        r = golden_uploaded.post("/chat", json={"question": "annual leave carry forward"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"answer", "model", "confidence", "sources"}
        assert body["answer"] == "GEMINI_ANSWER" and body["model"] == "gemini-2.0-flash"
        assert body["confidence"] in {"high", "medium", "low"}
        assert isinstance(body["sources"], list) and all(isinstance(s, str) for s in body["sources"])

    def test_sources_are_unique_filenames_of_retrieved_chunks(self, golden_uploaded, gemini_ok):
        body = golden_uploaded.post("/chat", json={"question": "annual leave"}).json()
        assert sorted(body["sources"]) == sorted(f"{s}.pdf" for s in GOLDEN_DOCS)
        assert len(body["sources"]) == len(set(body["sources"]))

    def test_sources_deduplicated_when_one_document_yields_several_chunks(self, client, embedder, gemini_ok):
        from tests.pdfutil import numbered_words
        words = numbered_words(1400).split()  # -> two chunks from long.pdf
        upload(client, "long.pdf", make_pdf_bytes([" ".join(words[i:i + 350]) for i in range(0, 1400, 350)]))
        upload(client, "enterprise_leave_policy.pdf")
        captured = []
        real_retrieve = main.retrieve

        def recording_retrieve(q):
            out = real_retrieve(q)
            captured.extend(out)
            return out

        with mock.patch.object(main, "retrieve", side_effect=recording_retrieve):
            body = client.post("/chat", json={"question": "w10 w20 w900 w1000"}).json()
        retrieved = [c["source"] for c in captured]
        assert retrieved.count("long.pdf") >= 2, "precondition: duplicate sources were retrieved"
        assert sorted(body["sources"]) == ["enterprise_leave_policy.pdf", "long.pdf"]

    def test_retrieved_context_with_citations_reaches_llm(self, golden_uploaded, gemini_ok):
        golden_uploaded.post("/chat", json={"question": "rubisco Calvin cycle stroma"})
        contents = gemini_ok.call_args.kwargs["contents"]
        assert "[Source: study_photosynthesis_notes.pdf]" in contents
        assert contents.endswith("Question: rubisco Calvin cycle stroma")

    @pytest.mark.parametrize("mode", ["enterprise", "legal", "medical", "study"])
    def test_all_four_modes_end_to_end(self, golden_uploaded, gemini_ok, mode):
        r = golden_uploaded.post("/chat", json={"question": "q", "mode": mode})
        assert r.status_code == 200
        assert gemini_ok.call_args.kwargs["contents"].startswith(llm.SYSTEM_PROMPTS[mode])
        with open("data/audit_log.csv") as f:
            assert f",{mode},q," in f.read()

    def test_mode_defaults_to_enterprise(self, golden_uploaded, gemini_ok):
        golden_uploaded.post("/chat", json={"question": "q"})
        assert gemini_ok.call_args.kwargs["contents"].startswith(llm.SYSTEM_PROMPTS["enterprise"])

    def test_fallback_through_api(self, golden_uploaded, gemini_down, groq_ok):
        body = golden_uploaded.post("/chat", json={"question": "q", "mode": "legal"}).json()
        assert (body["answer"], body["model"]) == ("GROQ_ANSWER", "groq-llama-3.3")

    def test_chat_result_matches_llm_confidence(self, golden_uploaded, gemini_ok):
        with mock.patch.object(main, "ask", wraps=main.ask) as spy:
            body = golden_uploaded.post("/chat", json={"question": "q"}).json()
        chunks = spy.call_args.args[0]
        assert body["confidence"] == llm.get_confidence(chunks)

    def test_both_llms_down_still_returns_200_with_response_contract(self, golden_uploaded,
                                                                     gemini_down, groq_down):
        r = golden_uploaded.post("/chat", json={"question": "q"})
        assert r.status_code == 200
        assert set(r.json()) == {"answer", "model", "confidence", "sources"}


# ---------------------------------------------------------------- known defects (xfail strict)
class TestKnownDefects:
    # D-03 FIXED (Phase 1, P1A): strict xfail removed; original assertions unchanged.
    def test_upload_embeds_only_new_document(self, client, embedder):
        upload(client, "enterprise_leave_policy.pdf")
        before = embedder.count
        upload(client, "legal_service_agreement.pdf")
        assert embedder.count - before == 1  # one new single-chunk document

    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-04: LLM outage is reported as 'No documents indexed yet'")
    def test_llm_outage_is_not_reported_as_missing_documents(self, golden_uploaded, gemini_down, groq_down):
        assert golden_uploaded.post("/chat", json={"question": "q"}).json()["answer"] != NO_DOCS["answer"]

    # D-05 FIXED (Phase 1): strict xfail removed; original assertions unchanged.
    def test_malformed_upload_does_not_break_later_uploads(self, client, embedder):
        upload(client, "corrupt.pdf", b"not a pdf")
        assert upload(client, "enterprise_leave_policy.pdf").status_code == 200

    # D-08 FIXED (Phase 1): strict xfail removed; original assertions unchanged.
    def test_upload_cannot_write_outside_pdf_folder(self, client, embedder, isolated_workdir):
        (isolated_workdir / "data" / "pdfs").mkdir(parents=True)
        upload(client, "../escaped.pdf", make_pdf_bytes(["escape attempt " * 10]))
        assert not (isolated_workdir / "data" / "escaped.pdf").exists()

    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-06: non-PDF files are saved and reported as indexed")
    def test_non_pdf_upload_is_not_reported_as_indexed(self, client, embedder):
        r = upload(client, "notes.txt", b"hello " * 50, "text/plain")
        assert "indexed successfully" not in r.json().get("message", "")

    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-09: blank/image-only PDF is reported as indexed successfully")
    def test_blank_pdf_upload_is_not_reported_as_indexed(self, client, embedder):
        r = upload(client, "scan.pdf", make_pdf_bytes([""]))
        assert "indexed successfully" not in r.json().get("message", "")
