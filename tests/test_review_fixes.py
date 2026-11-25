"""Regression tests for the d7df67d1 review: upload snapshot consistency, PyMuPDF thread model,
concurrency-limit validation, and chat responsiveness while ingestion is busy."""
import asyncio
import hashlib
import json
import os
import threading
import time
from unittest import mock

import httpx
import pytest

from backend import index_store, ingest, main, pdf_runtime, structure_chunker
from tests.pdfutil import golden_text, make_pdf_bytes, write_pdf

A = make_pdf_bytes([golden_text("legal_service_agreement")])
B = make_pdf_bytes([golden_text("medical_blood_report")])


def manifest_doc(name):
    m = json.loads(open(index_store.MANIFEST_PATH).read())
    return next(d for d in m["documents"] if d["source"] == name)


def test_fingerprint_and_parse_use_the_same_bytes_snapshot(isolated_workdir, embedder, monkeypatch):
    path = write_pdf(isolated_workdir / "data/pdfs/doc.pdf", [golden_text("legal_service_agreement")])
    original = path.read_bytes()  # PDF bytes are not deterministic across generations, so hash the file
    real = ingest.chunk_document

    def swap_then_chunk(p, data=None):
        path.write_bytes(B)  # another writer replaces the file after it was read and hashed
        return real(p, data)
    monkeypatch.setattr(ingest, "chunk_document", swap_then_chunk)
    ingest.build_index()
    gen = index_store.load_generation()
    assert manifest_doc("doc.pdf")["file_sha256"] == hashlib.sha256(original).hexdigest()
    assert [t for _, t in gen.chunks] == ingest.chunk_text(ingest.parse_pdf(original))  # chunks match hashed bytes
    assert path.read_bytes() == B  # the swap really happened


def test_same_name_upload_replacement_waits_for_in_flight_build(isolated_workdir, embedder, monkeypatch):
    folder = isolated_workdir / "data/pdfs"
    folder.mkdir(parents=True)
    events, real_chunk, real_replace = [], ingest.chunk_document, os.replace

    def slow_chunk(p, data=None):
        time.sleep(0.3)
        return real_chunk(p, data)

    def recording_replace(src, dst):
        if ".upload-" in str(src):
            events.append(("replace", dst, time.perf_counter()))
        return real_replace(src, dst)
    monkeypatch.setattr(ingest, "chunk_document", slow_chunk)
    monkeypatch.setattr(os, "replace", recording_replace)

    def upload(content, tag):
        staged = folder / f".upload-{tag}.partial"
        staged.write_bytes(content)
        ingest.replace_and_build(str(staged), str(folder / "doc.pdf"))
        events.append(("built", tag, time.perf_counter()))
    t1 = threading.Thread(target=upload, args=(A, "a"))
    t1.start()
    time.sleep(0.05)  # A holds the build lock (slow parse in progress)
    t2 = threading.Thread(target=upload, args=(B, "b"))
    t2.start()
    t1.join(10)
    t2.join(10)
    assert not t1.is_alive() and not t2.is_alive(), "deadlock"
    built_a = next(t for kind, tag, t in events if kind == "built" and tag == "a")
    replaces = [t for kind, _, t in events if kind == "replace"]
    assert len(replaces) == 2 and replaces[1] > built_a, "second replacement happened during the first build"
    gen = index_store.load_generation()
    assert manifest_doc("doc.pdf")["file_sha256"] == hashlib.sha256((folder / "doc.pdf").read_bytes()).hexdigest()
    assert [t for _, t in gen.chunks] == ingest.chunk_text(ingest.parse_pdf(str(folder / "doc.pdf")))


def test_pymupdf_only_runs_on_its_dedicated_thread(tmp_path, monkeypatch):
    pdf = write_pdf(tmp_path / "x.pdf", [golden_text("study_photosynthesis_notes")])
    seen, real_open = set(), ingest.fitz.open

    def recording_open(*a, **k):
        seen.add(threading.current_thread().name)
        return real_open(*a, **k)
    monkeypatch.setattr(ingest.fitz, "open", recording_open)
    threads = [threading.Thread(target=f) for f in
               [lambda: ingest.parse_pdf(str(pdf)), lambda: ingest.parse_pdf(pdf.read_bytes()),
                lambda: structure_chunker.chunk_pdf_structured(str(pdf))] * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen and all(n.startswith("pymupdf-worker") for n in seen), seen


@pytest.mark.parametrize("raw", ["0", "-1", "65", "abc", ""])
def test_invalid_concurrency_limits_are_rejected(monkeypatch, raw):
    monkeypatch.setenv("DOCINTEL_MAX_CONCURRENT_CHAT", raw)
    with pytest.raises(ValueError):
        main._limit("DOCINTEL_MAX_CONCURRENT_CHAT", 8)


def test_valid_concurrency_limit(monkeypatch):
    monkeypatch.setenv("DOCINTEL_MAX_CONCURRENT_CHAT", "12")
    assert main._limit("DOCINTEL_MAX_CONCURRENT_CHAT", 8) == 12


def test_chat_proceeds_while_ingestion_occupies_the_pdf_thread(isolated_workdir, embedder, strict_external_clients, monkeypatch):
    write_pdf(isolated_workdir / "data/pdfs/a.pdf", [golden_text("enterprise_leave_policy")])
    ingest.build_index()
    strict_external_clients["gemini"].models.generate_content.side_effect = lambda **kw: mock.Mock(text="FAST")
    real_parse = ingest._parse_pdf
    monkeypatch.setattr(ingest, "_parse_pdf", lambda pdf: (time.sleep(0.6), real_parse(pdf))[1])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
            up = asyncio.create_task(c.post("/upload", files={"file": ("b.pdf", B, "application/pdf")}))
            await asyncio.sleep(0.05)
            t = time.perf_counter()
            r = await c.post("/chat", json={"question": "annual leave"})
            chat_s = time.perf_counter() - t
            return r, chat_s, await up
    r, chat_s, up = asyncio.run(scenario())
    assert r.json()["answer"] == "FAST" and chat_s < 0.4, f"chat waited {chat_s:.2f}s behind ingestion"
    assert up.status_code == 200
