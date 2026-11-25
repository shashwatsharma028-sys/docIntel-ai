"""Phase 1 / P1B targeted tests: version-aware in-memory generation cache in the retriever."""
import os
import pickle
import shutil
import threading
import time

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import index_store, ingest, main, retriever
from tests.fakes import embed_text
from tests.pdfutil import golden_text, make_pdf_bytes, write_pdf

GOLDEN = ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report",
          "study_photosynthesis_notes"]
QUERIES = ["annual leave carry forward", "automatic renewal termination", "haemoglobin ferritin",
           "calvin cycle rubisco", "probation eligibility", "indemnification cap", "fasting glucose",
           "chlorophyll thylakoid", "arbitration delaware", "LDL cholesterol"]


@pytest.fixture
def folder(isolated_workdir):
    p = isolated_workdir / "data" / "pdfs"
    p.mkdir(parents=True)
    return p


@pytest.fixture
def golden(folder, embedder):
    for stem in GOLDEN:
        write_pdf(folder / f"{stem}.pdf", [golden_text(stem)])
    ingest.build_index()
    return folder


def loads():
    return index_store.COUNTERS["index_disk_loads"]


def original_retrieve(query, top_k=4):
    """The Phase 0 algorithm (disk load per query), with the D-01 guard, as an oracle."""
    index = faiss.read_index(retriever.INDEX_PATH)
    with open(retriever.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    _, idx = index.search(np.array([embed_text(query)], dtype="float32"), top_k)
    return [{"source": chunks[i][0], "text": chunks[i][1]} for i in idx[0] if 0 <= i < len(chunks)]


# ----------------------------------------------------------------------------- reuse
class TestReuse:
    def test_index_is_loaded_once_for_repeated_queries(self, golden):
        start = loads()
        for q in QUERIES:
            retriever.retrieve(q)
        assert loads() - start == 1

    def test_results_identical_to_uncached_original_algorithm(self, golden):
        for q in QUERIES:
            for k in (1, 4, 10):
                assert retriever.retrieve(q, top_k=k) == original_retrieve(q, top_k=k)

    def test_cold_and_warm_results_identical(self, golden):
        warm = [retriever.retrieve(q) for q in QUERIES]
        cold = []
        for q in QUERIES:
            retriever.clear_cache()
            cold.append(retriever.retrieve(q))
        assert cold == warm

    def test_ties_resolve_identically(self, folder, embedder):
        for name in ("a.pdf", "b.pdf", "c.pdf"):  # identical content -> equal distances
            write_pdf(folder / name, [golden_text("legal_service_agreement")])
        ingest.build_index()
        assert retriever.retrieve("termination") == original_retrieve("termination")
        assert [r["source"] for r in retriever.retrieve("termination", top_k=3)] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_cache_info_reports_resident_size(self, golden):
        retriever.retrieve("x")
        info = retriever.cache_info()
        entry = [e for e in info["entries"] if e["directory"] == os.path.realpath("data/index")][0]
        gen = retriever.current_generation()
        text_bytes = sum(len(s) + len(t) for s, t in gen.chunks)
        assert entry["resident_bytes"] == gen.index.ntotal * gen.index.d * 4 + text_bytes
        assert entry["n_chunks"] == 4 and entry["generation"] == 1


# ----------------------------------------------------------------------------- invalidation
class TestInvalidation:
    @pytest.fixture
    def client(self):
        return TestClient(main.app, raise_server_exceptions=False)

    def test_upload_invalidates(self, client, embedder, gemini_ok):
        client.post("/upload", files={"file": ("a.pdf", make_pdf_bytes([golden_text("legal_service_agreement")]), "application/pdf")})
        assert client.post("/chat", json={"question": "q"}).json()["sources"] == ["a.pdf"]
        client.post("/upload", files={"file": ("b.pdf", make_pdf_bytes([golden_text("medical_blood_report")]), "application/pdf")})
        assert sorted(client.post("/chat", json={"question": "q"}).json()["sources"]) == ["a.pdf", "b.pdf"]

    def test_replacement_invalidates(self, folder, embedder):
        write_pdf(folder / "doc.pdf", [golden_text("legal_service_agreement")])
        ingest.build_index()
        assert "Delaware" in retriever.retrieve("x")[0]["text"]
        write_pdf(folder / "doc.pdf", [golden_text("study_photosynthesis_notes")])
        ingest.build_index()
        assert "rubisco" in retriever.retrieve("x")[0]["text"]

    def test_deletion_to_empty_invalidates(self, golden):
        retriever.retrieve("x")
        for stem in GOLDEN:
            os.remove(golden / f"{stem}.pdf")
        ingest.build_index()
        with pytest.raises(FileNotFoundError):
            retriever.retrieve("x")

    def test_deletion_of_one_document_invalidates(self, golden):
        retriever.retrieve("x")
        os.remove(golden / "medical_blood_report.pdf")
        ingest.build_index()
        assert "medical_blood_report.pdf" not in {r["source"] for r in retriever.retrieve("x", top_k=10)}

    def test_unchanged_reupload_keeps_cache_warm(self, golden):
        retriever.retrieve("x")
        start = loads()
        write_pdf(golden / "legal_service_agreement.pdf", [golden_text("legal_service_agreement")])
        ingest.build_index()
        retriever.retrieve("x")
        assert loads() == start

    def test_first_query_after_invalidation_reloads_exactly_once(self, golden):
        retriever.retrieve("x")
        write_pdf(golden / "extra.pdf", [golden_text("legal_service_agreement") + " extra clause"])
        ingest.build_index()
        start, reloads = loads(), retriever.STATS["reloads"]
        for q in QUERIES:
            retriever.retrieve(q)
        assert loads() - start == 1 and retriever.STATS["reloads"] == reloads + 1

    def test_legacy_writer_replacing_files_is_noticed(self, isolated_workdir, embedder):
        """Files replaced by a writer that does not know about the cache (Phase 0 test pattern)."""
        def write(texts):
            os.makedirs("data/index", exist_ok=True)
            index = faiss.IndexFlatL2(256)
            index.add(np.stack([embed_text(t) for t in texts]))
            faiss.write_index(index, retriever.INDEX_PATH + ".new")
            with open(retriever.CHUNKS_PATH + ".new", "wb") as f:
                pickle.dump([(f"{t}.pdf", t) for t in texts], f)
            os.replace(retriever.INDEX_PATH + ".new", retriever.INDEX_PATH)
            os.replace(retriever.CHUNKS_PATH + ".new", retriever.CHUNKS_PATH)
        write(["alpha"])
        assert retriever.retrieve("alpha", top_k=1)[0]["source"] == "alpha.pdf"
        write(["beta", "gamma"])
        assert {r["source"] for r in retriever.retrieve("beta", top_k=2)} == {"beta.pdf", "gamma.pdf"}


# ----------------------------------------------------------------------------- isolation and bounds
class TestIsolationAndBounds:
    def _make(self, root, stem):
        d = root / stem
        write_pdf(d / "data" / "pdfs" / f"{stem}.pdf", [golden_text(stem)])
        os.chdir(d)
        ingest.build_index()
        return d

    def test_directories_never_share_entries(self, isolated_workdir, embedder):
        a = self._make(isolated_workdir, "legal_service_agreement")
        b = self._make(isolated_workdir, "medical_blood_report")
        for _ in range(3):
            os.chdir(a)
            assert retriever.retrieve("x")[0]["source"] == "legal_service_agreement.pdf"
            os.chdir(b)
            assert retriever.retrieve("x")[0]["source"] == "medical_blood_report.pdf"

    def test_lru_bound_on_number_of_cached_generations(self, isolated_workdir, embedder, monkeypatch):
        monkeypatch.setattr(retriever, "MAX_CACHED_GENERATIONS", 1)
        retriever.clear_cache()
        dirs = [self._make(isolated_workdir, s) for s in GOLDEN[:3]]
        evictions = retriever.STATS["evictions"]
        for d in dirs:
            os.chdir(d)
            retriever.retrieve("x")
        info = retriever.cache_info()
        assert len(info["entries"]) == 1 and info["entries"][0]["directory"] == os.path.realpath(dirs[-1] / "data/index")
        assert retriever.STATS["evictions"] - evictions == 2  # first entry replaced the cleared state


# ----------------------------------------------------------------------------- safety
class TestSafety:
    def test_malicious_legacy_pickle_is_rejected_not_executed(self, isolated_workdir, embedder, tmp_path):
        marker = tmp_path / "pwned"

        class Evil:
            def __reduce__(self):
                return (open, (str(marker), "w"))
        os.makedirs("data/index")
        index = faiss.IndexFlatL2(256)
        index.add(embed_text("a").reshape(1, -1))
        faiss.write_index(index, retriever.INDEX_PATH)
        with open(retriever.CHUNKS_PATH, "wb") as f:
            pickle.dump([("a.pdf", Evil())], f)
        with pytest.raises(pickle.UnpicklingError):
            retriever.retrieve("a")
        assert not marker.exists()

    def test_misaligned_legacy_files_are_rejected(self, isolated_workdir, embedder):
        os.makedirs("data/index")
        index = faiss.IndexFlatL2(256)
        index.add(np.stack([embed_text("a"), embed_text("b")]))
        faiss.write_index(index, retriever.INDEX_PATH)
        with open(retriever.CHUNKS_PATH, "wb") as f:
            pickle.dump([("a.pdf", "a")], f)
        with pytest.raises(index_store.IndexIntegrityError):
            retriever.retrieve("a")

    def test_integrity_failure_maps_to_existing_chat_error_response(self, golden, gemini_ok):
        other = faiss.IndexFlatL2(256)
        other.add(np.stack([embed_text(f"x{i}") for i in range(4)]))
        faiss.write_index(other, retriever.INDEX_PATH + ".x")
        os.replace(retriever.INDEX_PATH + ".x", retriever.INDEX_PATH)
        body = TestClient(main.app).post("/chat", json={"question": "q"}).json()
        assert body == {"answer": "No documents indexed yet. Please upload a PDF first.",
                        "model": "", "confidence": "low", "sources": []}


# ----------------------------------------------------------------------------- concurrency
def test_concurrent_readers_never_observe_mixed_generations(folder, embedder):
    """Soak/smoke test: readers loop while a writer alternates between two corpora. Every result
    list must come from exactly one corpus state and every (source, text) pair must be real.
    NOTE: the swap window is tiny, so this test alone cannot prove locking; the deterministic
    test_reader_during_mid_swap_sees_complete_generation does."""
    states = {
        "A": {f"A{i}_{s}.pdf": golden_text(s) for i, s in enumerate(GOLDEN)},
        "B": {f"B{i}_{s}.pdf": golden_text(s) + " revision B" for i, s in enumerate(GOLDEN[:3])},
    }
    pairs = {(n, " ".join(t.split())) for st in states.values() for n, t in st.items()}

    def apply(state):
        for p in folder.iterdir():
            p.unlink()
        for name, text in states[state].items():
            write_pdf(folder / name, [text])
        ingest.build_index()

    apply("A")
    stop, errors, observed = threading.Event(), [], set()

    def reader():
        while not stop.is_set():
            try:
                res = retriever.retrieve("leave termination glucose chlorophyll", top_k=10)
            except FileNotFoundError:
                continue  # between unlink and publish nothing is torn; the old generation is
            except Exception as e:  # pragma: no cover    # still served or the index is absent
                errors.append(repr(e))
                return
            prefixes = {r["source"][0] for r in res}
            if len(prefixes) != 1 or not all((r["source"], r["text"]) in pairs for r in res):
                errors.append(f"mixed or misaligned result: {res}")
                return
            observed.add((prefixes.pop(), len(res)))

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    deadline = time.time() + 2.0
    flips = 0
    while time.time() < deadline:
        apply("B" if flips % 2 == 0 else "A")
        flips += 1
    stop.set()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert {"A", "B"} <= {p for p, _ in observed} and flips >= 4
    assert observed <= {("A", 4), ("B", 3)}


def test_reader_during_mid_swap_sees_complete_generation(folder, embedder, monkeypatch):
    """Deterministic interleaving: pause the writer right after faiss.index is swapped in
    (chunks.pkl / manifest.json still old) and run a cold reader at exactly that moment.
    The shared publish lock must make the reader wait and then see the complete new generation;
    it must neither return mixed data nor fail."""
    for stem in GOLDEN:
        write_pdf(folder / f"A_{stem}.pdf", [golden_text(stem)])
    ingest.build_index()
    for p in folder.iterdir():
        p.unlink()
    for stem in GOLDEN[:3]:
        write_pdf(folder / f"B_{stem}.pdf", [golden_text(stem) + " revision B"])

    mid_swap, reader_done, outcome = threading.Event(), threading.Event(), {}
    real_replace = os.replace

    def pausing_replace(src, dst):
        real_replace(src, dst)
        if dst == index_store.INDEX_PATH and ".tmp-" in str(src):
            mid_swap.set()
            reader_done.wait(timeout=0.5)  # a correct reader is blocked by the lock -> times out

    def reader():
        mid_swap.wait(timeout=10)
        retriever.clear_cache()
        try:
            outcome["result"] = retriever.retrieve("leave termination glucose", top_k=10)
        except Exception as e:
            outcome["error"] = e
        reader_done.set()

    t = threading.Thread(target=reader)
    t.start()
    monkeypatch.setattr(os, "replace", pausing_replace)
    ingest.build_index()
    monkeypatch.setattr(os, "replace", real_replace)
    t.join(timeout=10)
    assert "error" not in outcome, f"reader failed mid-swap: {outcome.get('error')!r}"
    assert sorted(r["source"] for r in outcome["result"]) == sorted(f"B_{s}.pdf" for s in GOLDEN[:3])
