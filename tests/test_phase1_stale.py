"""Phase 1 targeted tests for D-09a: blank/empty corpus must not leave stale searchable content."""
import os

import pytest
from fastapi.testclient import TestClient

from backend import ingest, main, retriever
from tests.pdfutil import golden_text, make_pdf_bytes, write_pdf

NO_DOCS = "No documents indexed yet. Please upload a PDF first."


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def test_blank_replacement_of_only_document_removes_its_content(client, embedder, gemini_ok):
    client.post("/upload", files={"file": ("doc.pdf", make_pdf_bytes([golden_text("medical_blood_report")]), "application/pdf")})
    assert client.post("/chat", json={"question": "haemoglobin"}).json()["sources"] == ["doc.pdf"]
    client.post("/upload", files={"file": ("doc.pdf", make_pdf_bytes([""]), "application/pdf")})
    body = client.post("/chat", json={"question": "haemoglobin"}).json()
    assert body["answer"] == NO_DOCS and body["sources"] == []
    with pytest.raises(Exception):  # same contract as Phase 0 test_missing_index_raises
        retriever.retrieve("haemoglobin")


def test_blank_replacement_among_others_removes_only_that_document(client, embedder, gemini_ok):
    for stem in ("medical_blood_report", "legal_service_agreement"):
        client.post("/upload", files={"file": (f"{stem}.pdf", make_pdf_bytes([golden_text(stem)]), "application/pdf")})
    client.post("/upload", files={"file": ("medical_blood_report.pdf", make_pdf_bytes([""]), "application/pdf")})
    results = retriever.retrieve("haemoglobin ferritin anaemia", top_k=4)
    assert {r["source"] for r in results} == {"legal_service_agreement.pdf"}


def test_removing_all_documents_then_rebuilding_clears_index(isolated_workdir, embedder):
    folder = isolated_workdir / "data" / "pdfs"
    write_pdf(folder / "a.pdf", [golden_text("study_photosynthesis_notes")])
    ingest.build_index()
    os.remove(folder / "a.pdf")
    ingest.build_index()
    with pytest.raises(Exception):  # same contract as Phase 0 test_missing_index_raises
        retriever.retrieve("chlorophyll")
