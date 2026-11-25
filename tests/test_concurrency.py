"""Event-loop responsiveness, bounded provider concurrency and consistent concurrent uploads."""
import asyncio
import csv
import threading
import time
from unittest import mock

import httpx
import pytest

from backend import index_store, ingest, main
from tests.pdfutil import golden_text, make_pdf_bytes

STEMS = ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report", "study_photosynthesis_notes"]


class SlowLLM:
    """Simulated blocking provider: tracks peak concurrency."""
    def __init__(self, seconds):
        self.seconds, self.active, self.peak, self.lock = seconds, 0, 0, threading.Lock()

    def __call__(self, **kw):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.seconds)
        with self.lock:
            self.active -= 1
        return mock.Mock(text="SIMULATED_ANSWER")


@pytest.fixture
def indexed(isolated_workdir, embedder):
    folder = isolated_workdir / "data" / "pdfs"
    folder.mkdir(parents=True)
    from tests.pdfutil import write_pdf
    for s in STEMS:
        write_pdf(folder / f"{s}.pdf", [golden_text(s)])
    ingest.build_index()


def run(coro):
    return asyncio.run(coro)


def asgi_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t")


def test_health_responsive_while_chat_blocks_in_provider(indexed, strict_external_clients):
    strict_external_clients["gemini"].models.generate_content.side_effect = SlowLLM(0.5)

    async def scenario():
        async with asgi_client() as c:
            chat = asyncio.create_task(c.post("/chat", json={"question": "q"}))
            t = time.perf_counter()  # timed from BEFORE yielding: a blocked loop delays everything after this
            await asyncio.sleep(0.02)
            h = await c.get("/health")
            health_ms = (time.perf_counter() - t) * 1000
            r = await chat
            return h, health_ms, r
    h, health_ms, r = run(scenario())
    assert h.json() == {"status": "ok"} and health_ms < 200, f"/health took {health_ms:.0f} ms"
    assert r.json()["answer"] == "SIMULATED_ANSWER"


def test_provider_concurrency_is_bounded(indexed, strict_external_clients, monkeypatch):
    llm = SlowLLM(0.1)
    strict_external_clients["gemini"].models.generate_content.side_effect = llm
    monkeypatch.setattr(main, "MAX_CONCURRENT_CHAT", 3, raising=False)

    async def scenario():
        async with asgi_client() as c:
            return await asyncio.gather(*[c.post("/chat", json={"question": f"q{i}"}) for i in range(12)])
    rs = run(scenario())
    assert all(r.status_code == 200 and r.json()["answer"] == "SIMULATED_ANSWER" for r in rs)
    assert 1 < llm.peak <= 3


def test_audit_rows_do_not_interleave_under_concurrency(indexed, strict_external_clients):
    strict_external_clients["gemini"].models.generate_content.side_effect = SlowLLM(0.01)

    async def scenario():
        async with asgi_client() as c:
            return await asyncio.gather(*[c.post("/chat", json={"question": f'q{i}, "x"\nline'}) for i in range(24)])
    run(scenario())
    with open("data/audit_log.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 24 and all(len(r) == 5 for r in rows)
    assert sorted(r[2] for r in rows) == sorted(f'q{i}, "x"\nline' for i in range(24))


def test_concurrent_uploads_incl_duplicate_names_publish_coherent_index(isolated_workdir, embedder):
    files = [(f"{s}.pdf", make_pdf_bytes([golden_text(s)])) for s in STEMS]
    files += [("dup.pdf", make_pdf_bytes([golden_text("legal_service_agreement") + " version one"])),
              ("dup.pdf", make_pdf_bytes([golden_text("medical_blood_report") + " version two"]))]

    async def scenario():
        async with asgi_client() as c:
            return await asyncio.gather(*[c.post("/upload", files={"file": (n, b, "application/pdf")}) for n, b in files])
    rs = run(scenario())
    assert all(r.status_code == 200 for r in rs)
    gen = index_store.load_generation()  # coherent generation (identity checks + count alignment)
    on_disk = {p.name: ingest.chunk_text(ingest.parse_pdf(str(p))) for p in (isolated_workdir / "data/pdfs").iterdir()}
    assert sorted(gen.chunks) == sorted((n, t) for n, ts in on_disk.items() for t in ts)  # index == final folder
    assert len(on_disk) == 5


def test_bounds_work_across_event_loops(indexed, strict_external_clients):
    """Regression: a module-level asyncio.Semaphore raised 'bound to a different event loop'."""
    strict_external_clients["gemini"].models.generate_content.side_effect = SlowLLM(0.01)

    async def burst():
        async with asgi_client() as c:
            return await asyncio.gather(*[c.post("/chat", json={"question": "q"}) for _ in range(12)])
    for _ in range(3):  # three separate event loops, each with contention
        assert all(r.status_code == 200 for r in run(burst()))
