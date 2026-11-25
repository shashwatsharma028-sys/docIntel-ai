"""Phase 1 / P1A targeted tests: incremental indexing, embedding reuse, atomic publication."""
import json
import os
import pickle
import shutil
import threading

import faiss
import numpy as np
import pytest

from backend import embedding_cache, index_store, ingest, retriever
from tests.fakes import embed_text
from tests.pdfutil import golden_text, numbered_words, write_pdf

GOLDEN = ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report",
          "study_photosynthesis_notes"]


@pytest.fixture
def folder(isolated_workdir):
    p = isolated_workdir / "data" / "pdfs"
    p.mkdir(parents=True)
    return p


def add(folder, name, stem=None, pages=None):
    return write_pdf(folder / name, pages if pages is not None else [golden_text(stem)])


def published():
    index = faiss.read_index(ingest.INDEX_PATH)
    with open(ingest.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(index_store.MANIFEST_PATH) as f:
        manifest = json.load(f)
    return index, chunks, manifest


def assert_aligned(fake=embed_text):
    index, chunks, manifest = published()
    assert index.ntotal == len(chunks) == manifest["n_chunks"]
    stored = index.reconstruct_n(0, index.ntotal)
    for i, (_, text) in enumerate(chunks):
        assert np.array_equal(stored[i], fake(text)), f"row {i} is not the embedding of chunk {i}"
    offset = 0
    for doc in manifest["documents"]:
        assert all(src == doc["source"] for src, _ in chunks[offset:offset + doc["n_chunks"]])
        offset += doc["n_chunks"]
    assert offset == len(chunks)
    return index, chunks, manifest


def long_pages(n_words=1400, mutate_at=None):
    words = numbered_words(n_words).split()
    if mutate_at is not None:
        words[mutate_at] = "CHANGED"
    return [" ".join(words[i:i + 350]) for i in range(0, n_words, 350)]


# ----------------------------------------------------------------------------- reuse
class TestIncrementalReuse:
    def test_sequential_uploads_embed_each_chunk_once(self, folder, embedder):
        for i, stem in enumerate(GOLDEN, 1):
            add(folder, f"{stem}.pdf", stem)
            stats = ingest.build_index()
            assert stats["embeddings_computed"] == 1 and stats["cache_hits"] == i - 1
            assert stats["documents_parsed"] == 1 and stats["documents_reused"] == i - 1
        assert embedder.count == len(GOLDEN)
        assert_aligned()

    def test_identical_reupload_embeds_nothing_and_keeps_generation(self, folder, embedder):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        before = index_store.signature()
        count = embedder.count
        add(folder, "a.pdf", "legal_service_agreement")  # same bytes rewritten
        stats = ingest.build_index()
        assert stats["unchanged"] and not stats["published"]
        assert embedder.count == count
        assert index_store.signature() == before  # no new generation, readers stay warm

    def test_modified_document_embeds_only_changed_chunks(self, folder, embedder):
        add(folder, "long.pdf", pages=long_pages())  # chunks [0,800) [700,1400)
        ingest.build_index()
        count = embedder.count
        add(folder, "long.pdf", pages=long_pages(mutate_at=1300))  # only chunk 2 changes
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 1 and stats["cache_hits"] == 1
        assert embedder.count == count + 1
        _, chunks, _ = assert_aligned()
        assert not any("w1300" in t.split() for _, t in chunks), "stale chunk text still indexed"
        assert "CHANGED" in chunks[1][1].split() and "CHANGED" not in chunks[0][1].split()
        # exact-match retrieval of the new chunk (the hashed fake embedder cannot rank long
        # chunks by a few tokens: 700+ tokens saturate its 256 buckets)
        assert retriever.retrieve(chunks[1][1], top_k=1)[0] == {"source": "long.pdf", "text": chunks[1][1]}

    def test_same_name_replacement_removes_old_content(self, folder, embedder):
        add(folder, "doc.pdf", "medical_blood_report")
        ingest.build_index()
        add(folder, "doc.pdf", "study_photosynthesis_notes")
        ingest.build_index()
        _, chunks, _ = assert_aligned()
        assert [t.split()[:3] for _, t in chunks] == [golden_text("study_photosynthesis_notes").split()[:3]]

    def test_duplicate_content_keeps_distinct_source_identities(self, folder, embedder):
        add(folder, "copy_a.pdf", "enterprise_leave_policy")
        add(folder, "copy_b.pdf", "enterprise_leave_policy")
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 1 and stats["cache_hits"] == 1
        _, chunks, _ = assert_aligned()
        assert [s for s, _ in chunks] == ["copy_a.pdf", "copy_b.pdf"]
        top2 = retriever.retrieve("annual leave carried forward", top_k=2)
        assert {r["source"] for r in top2} == {"copy_a.pdf", "copy_b.pdf"}

    def test_removed_document_is_not_retrievable(self, folder, embedder):
        for stem in GOLDEN:
            add(folder, f"{stem}.pdf", stem)
        ingest.build_index()
        os.remove(folder / "medical_blood_report.pdf")
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 0
        assert_aligned()
        sources = {r["source"] for r in retriever.retrieve("haemoglobin ferritin anaemia glucose", top_k=4)}
        assert "medical_blood_report.pdf" not in sources and len(sources) == 3

    def test_rename_reuses_everything(self, folder, embedder):
        add(folder, "old.pdf", "legal_service_agreement")
        ingest.build_index()
        os.rename(folder / "old.pdf", folder / "new.pdf")
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 0 and stats["documents_reused"] == 1
        assert [s for s, _ in published()[1]] == ["new.pdf"]

    def test_mixed_existing_and_new_documents(self, folder, embedder):
        for stem in GOLDEN[:2]:
            add(folder, f"{stem}.pdf", stem)
        ingest.build_index()
        for stem in GOLDEN[2:]:
            add(folder, f"{stem}.pdf", stem)
        add(folder, "long.pdf", pages=long_pages())
        stats = ingest.build_index()
        assert (stats["embeddings_computed"], stats["cache_hits"], stats["documents_parsed"]) == (4, 2, 3)
        assert_aligned()

    def test_build_order_is_deterministic(self, folder, embedder):
        for name in ["zeta.pdf", "alpha.pdf", "mid.pdf"]:
            add(folder, name, "study_photosynthesis_notes")
        ingest.build_index()
        assert [s for s, _ in published()[1]] == ["alpha.pdf", "mid.pdf", "zeta.pdf"]


# ----------------------------------------------------------------------------- cache validity
class TestCacheValidity:
    def test_model_identity_change_invalidates_cache(self, folder, embedder, monkeypatch):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        monkeypatch.setattr(ingest, "EMBEDDING_MODEL", "some-other-embedding-model")
        add(folder, "b.pdf", "medical_blood_report")
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 2 and stats["cache_hits"] == 0
        assert stats["cache"]["load_status"] == "invalidated_namespace"
        assert embedder.calls[-1]["model"] == "some-other-embedding-model"

    def test_requested_dimensionality_is_part_of_namespace(self, folder, embedder, monkeypatch):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        monkeypatch.setattr(ingest, "EMBEDDING_OUTPUT_DIMENSIONALITY", 768)
        add(folder, "b.pdf", "medical_blood_report")
        assert ingest.build_index()["embeddings_computed"] == 2

    def test_provider_dimension_change_discards_cached_vectors(self, folder, strict_external_clients):
        from tests.fakes import FakeEmbedder
        fake = FakeEmbedder()
        strict_external_clients["ingest"].models.embed_content.side_effect = fake.embed_content
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()

        def half_dim(**kw):  # provider silently switches to 128-d vectors
            r = fake.embed_content(**kw)
            r.embeddings[0].values = r.embeddings[0].values[:128]
            return r
        strict_external_clients["ingest"].models.embed_content.side_effect = half_dim
        add(folder, "b.pdf", "medical_blood_report")
        stats = ingest.build_index()
        index, chunks, manifest = published()
        assert index.d == 128 and manifest["dim"] == 128 and index.ntotal == 2
        assert stats["embeddings_computed"] == 1 + 2  # 1 call detected the change, then both re-embedded

    def test_chunker_change_disables_document_reuse(self, folder, embedder, monkeypatch):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        monkeypatch.setattr(ingest, "CHUNKER_VERSION", 2)
        add(folder, "b.pdf", "medical_blood_report")
        stats = ingest.build_index()
        assert stats["documents_parsed"] == 2 and stats["embeddings_computed"] == 1  # same texts -> cache

    def test_stale_entries_are_bounded_live_entries_kept(self, folder, embedder, monkeypatch):
        monkeypatch.setenv("DOCINTEL_EMBED_CACHE_MAX_STALE_MB", "0")
        for stem in GOLDEN:
            add(folder, f"{stem}.pdf", stem)
        ingest.build_index()
        for stem in GOLDEN[1:]:
            os.remove(folder / f"{stem}.pdf")
        add(folder, "extra.pdf", "legal_service_agreement")
        stats = ingest.build_index()
        assert stats["cache"]["entries"] == 2 and stats["cache"]["evicted"] == 2
        # evicted content costs a re-embed if it comes back; live content never does
        add(folder, "medical_blood_report.pdf", "medical_blood_report")
        assert ingest.build_index()["embeddings_computed"] == 1

    def test_stale_budget_keeps_most_recent(self, folder, embedder, monkeypatch):
        per_entry_mb = 256 * 4 / (1024 * 1024)
        monkeypatch.setenv("DOCINTEL_EMBED_CACHE_MAX_STALE_MB", str(per_entry_mb * 1.5))  # room for 1 stale
        for stem in GOLDEN:
            add(folder, f"{stem}.pdf", stem)
            ingest.build_index()
        for stem in GOLDEN:
            os.remove(folder / f"{stem}.pdf")
        add(folder, "keep.pdf", pages=["completely different text about volcanoes and basalt lava flows"])
        assert ingest.build_index()["cache"]["entries"] == 2

    @pytest.mark.parametrize("payload", ["garbage", "object_array"])
    def test_corrupt_or_pickled_cache_is_discarded_safely(self, folder, embedder, payload):
        os.makedirs(embedding_cache.CACHE_DIR)
        if payload == "garbage":
            with open(embedding_cache.CACHE_PATH, "wb") as f:
                f.write(b"not an npz")
        else:
            np.savez(embedding_cache.CACHE_PATH, keys=np.array([object()], dtype=object))
        add(folder, "a.pdf", "legal_service_agreement")
        stats = ingest.build_index()
        assert stats["cache"]["load_status"] == "discarded_corrupt" and stats["embeddings_computed"] == 1
        assert_aligned()

    def test_cache_file_loads_without_pickle(self, folder, embedder):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        with np.load(embedding_cache.CACHE_PATH, allow_pickle=False) as z:
            assert z["vectors"].shape == (1, 256) and z["keys"].dtype.kind == "S"


# ----------------------------------------------------------------------------- failures
class TestFailureIsolation:
    def _three_new(self, folder):
        for stem in GOLDEN[1:]:
            add(folder, f"{stem}.pdf", stem)

    def test_partial_embedding_failure_keeps_previous_generation_and_progress(
            self, folder, embedder, strict_external_clients):
        add(folder, "enterprise_leave_policy.pdf", "enterprise_leave_policy")
        ingest.build_index()
        gen_before, sig_before = published()[2]["generation"], index_store.signature()
        self._three_new(folder)
        calls = {"n": 0}

        def fail_second(**kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated embedding outage")
            return embedder.embed_content(**kw)
        strict_external_clients["ingest"].models.embed_content.side_effect = fail_second
        with pytest.raises(RuntimeError, match="outage"):
            ingest.build_index()
        assert index_store.signature() == sig_before and published()[2]["generation"] == gen_before
        assert {r["source"] for r in retriever.retrieve("anything", top_k=4)} == {"enterprise_leave_policy.pdf"}

        strict_external_clients["ingest"].models.embed_content.side_effect = embedder.embed_content
        before = embedder.count
        stats = ingest.build_index()
        assert stats["embeddings_computed"] == 2  # the one embedded before the failure was kept
        assert embedder.count - before == 2
        assert_aligned()

    @pytest.mark.parametrize("fail_on", [1, 2, 3])
    def test_failed_publication_rolls_back_to_previous_generation(self, folder, embedder, monkeypatch, fail_on):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        files = {p: open(p, "rb").read() for p in (ingest.INDEX_PATH, ingest.CHUNKS_PATH, index_store.MANIFEST_PATH)}
        add(folder, "b.pdf", "medical_blood_report")
        real_replace, n = os.replace, {"i": 0}

        def failing_replace(src, dst):
            if ".tmp-" in str(src) and str(dst).startswith("data/index"):
                n["i"] += 1
                if n["i"] == fail_on:
                    raise OSError("simulated disk failure during publish")
            return real_replace(src, dst)
        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated"):
            ingest.build_index()
        monkeypatch.setattr(os, "replace", real_replace)
        assert {p: open(p, "rb").read() for p in files} == files, "previous generation was modified"
        gen = index_store.load_generation()
        assert [s for s, _ in gen.chunks] == ["a.pdf"]
        assert sorted(os.listdir("data/index")) == ["chunks.pkl", "faiss.index", "manifest.json"]

    def test_crash_mid_publish_is_detected_not_misread(self, folder, embedder):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        # Simulate a crash after the index rename but before chunks/manifest: foreign index file.
        other = faiss.IndexFlatL2(256)
        other.add(np.stack([embed_text("x"), embed_text("y")]))
        faiss.write_index(other, ingest.INDEX_PATH + ".crash")
        os.replace(ingest.INDEX_PATH + ".crash", ingest.INDEX_PATH)
        with pytest.raises(index_store.IndexIntegrityError):
            index_store.load_generation()
        # the next build does not trust the torn generation and repairs it from the corpus
        stats = ingest.build_index()
        assert stats["published"] and stats["documents_parsed"] == 1 and stats["embeddings_computed"] == 0
        assert_aligned()

    def test_torn_generation_with_equal_counts_is_detected(self, folder, embedder):
        """Same vector count, different content: only the manifest identity check can catch it."""
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        other = faiss.IndexFlatL2(256)
        other.add(embed_text("unrelated content").reshape(1, -1))
        faiss.write_index(other, ingest.INDEX_PATH + ".crash")
        os.replace(ingest.INDEX_PATH + ".crash", ingest.INDEX_PATH)
        with pytest.raises(index_store.IndexIntegrityError, match="does not belong"):
            index_store.load_generation()

    def test_restored_backup_with_new_inodes_is_accepted_by_content_hash(self, folder, embedder):
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        for p in (ingest.INDEX_PATH, ingest.CHUNKS_PATH):
            shutil.copy(p, p + ".restore")
            os.replace(p + ".restore", p)  # same bytes, new inode/mtime (e.g. backup restore)
        assert [s for s, _ in index_store.load_generation().chunks] == ["a.pdf"]

    def test_concurrent_builds_are_serialised_and_coherent(self, folder, embedder):
        for stem in GOLDEN:
            add(folder, f"{stem}.pdf", stem)
        errors = []

        def run():
            try:
                ingest.build_index()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)
        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert embedder.count == len(GOLDEN)  # serialised: later builds reuse the first one's work
        assert_aligned()


# ----------------------------------------------------------------------------- compatibility
class TestCompatibilityAndSafety:
    def test_legacy_layout_without_manifest_is_rebuilt(self, folder, embedder):
        add(folder, "a.pdf", "legal_service_agreement")
        os.makedirs("data/index")
        index = faiss.IndexFlatL2(256)
        index.add(embed_text("legacy").reshape(1, -1))
        faiss.write_index(index, ingest.INDEX_PATH)
        with open(ingest.CHUNKS_PATH, "wb") as f:
            pickle.dump([("legacy.pdf", "legacy")], f)
        stats = ingest.build_index()
        assert stats["published"] and stats["documents_parsed"] == 1
        assert [s for s, _ in assert_aligned()[1]] == ["a.pdf"]

    def test_malicious_metadata_is_never_executed(self, folder, embedder, tmp_path):
        marker = tmp_path / "pwned"

        class Evil:
            def __reduce__(self):
                return (open, (str(marker), "w"))
        add(folder, "a.pdf", "legal_service_agreement")
        ingest.build_index()
        with open(ingest.CHUNKS_PATH, "wb") as f:
            pickle.dump([("a.pdf", Evil())], f)
        with pytest.raises(pickle.UnpicklingError):
            index_store.loads_chunks(open(ingest.CHUNKS_PATH, "rb").read())
        assert index_store.load_metadata() is None
        ingest.build_index()
        assert not marker.exists()

    @pytest.mark.parametrize("obj", [{"a": 1}, [("a.pdf",)], [(1, "x")], [("a", "b", "c")]])
    def test_metadata_schema_is_validated(self, obj):
        with pytest.raises(index_store.IndexIntegrityError):
            index_store.loads_chunks(pickle.dumps(obj))

    def test_original_reader_can_still_read_new_artifacts(self, folder, embedder):
        """Rollback compatibility: the Phase 0 reader logic works on Phase 1 output."""
        for stem in GOLDEN:
            add(folder, f"{stem}.pdf", stem)
        ingest.build_index()
        index = faiss.read_index(ingest.INDEX_PATH)
        with open(ingest.CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        _, idx = index.search(embed_text("rubisco calvin cycle").reshape(1, -1), 1)
        assert chunks[idx[0][0]][0] == "study_photosynthesis_notes.pdf"


# ----------------------------------------------------------------------------- equivalence
def test_incremental_result_equals_clean_rebuild(folder, embedder, isolated_workdir):
    ops = [("add", "enterprise_leave_policy"), ("add", "legal_service_agreement"), ("long", None),
           ("add", "medical_blood_report"), ("modify_long", None), ("remove", "legal_service_agreement"),
           ("dup", "medical_blood_report"), ("add", "study_photosynthesis_notes")]
    for op, stem in ops:
        if op == "add":
            add(folder, f"{stem}.pdf", stem)
        elif op == "long":
            add(folder, "long.pdf", pages=long_pages())
        elif op == "modify_long":
            add(folder, "long.pdf", pages=long_pages(mutate_at=1300))
        elif op == "remove":
            os.remove(folder / f"{stem}.pdf")
        elif op == "dup":
            shutil.copy(folder / f"{stem}.pdf", folder / "copy_of_medical.pdf")
        ingest.build_index()
    inc_index, inc_chunks, _ = published()
    inc_results = [retriever.retrieve(q, top_k=4) for q in ("annual leave", "haemoglobin", "w1300 CHANGED",
                                                             "chlorophyll thylakoid", "termination notice")]

    clean = isolated_workdir / "clean"
    shutil.copytree(folder, clean / "data" / "pdfs")
    os.chdir(clean)
    ingest.build_index()
    clean_index, clean_chunks, _ = published()
    assert clean_chunks == inc_chunks
    assert np.array_equal(clean_index.reconstruct_n(0, clean_index.ntotal),
                          inc_index.reconstruct_n(0, inc_index.ntotal))
    assert [retriever.retrieve(q, top_k=4) for q in ("annual leave", "haemoglobin", "w1300 CHANGED",
                                                      "chlorophyll thylakoid", "termination notice")] == inc_results
