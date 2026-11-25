"""Opt-in batched embedding (DOCINTEL_EMBED_BATCH_SIZE > 1); the single-item default is unchanged."""
import json
import math
import pickle
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from backend import index_store, ingest
from tests.fakes import FakeEmbedder, embed_text
from tests.pdfutil import golden_text, numbered_words, write_pdf

STEMS = ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report", "study_photosynthesis_notes"]


@pytest.fixture
def corpus(isolated_workdir):
    folder = isolated_workdir / "data" / "pdfs"
    for s in STEMS:
        write_pdf(folder / f"{s}.pdf", [golden_text(s)])
    words = numbered_words(3000).split()
    write_pdf(folder / "long.pdf", [" ".join(words[i:i + 300]) for i in range(0, 3000, 300)])  # 5 chunks
    return folder


def published():
    idx = faiss.read_index(ingest.INDEX_PATH)
    with open(ingest.CHUNKS_PATH, "rb") as f:
        return pickle.load(f), idx.reconstruct_n(0, idx.ntotal)


def install(strict_external_clients, fake):
    strict_external_clients["ingest"].models.embed_content.side_effect = fake


def test_invalid_batch_size_rejected(monkeypatch):
    for bad in ("0", "101"):
        monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", bad)
        with pytest.raises(ValueError):
            ingest.embed_batch_size()


@pytest.mark.parametrize("bs", [2, 4, 8, 32])
def test_batched_index_identical_to_single_item_and_requests_bounded(corpus, strict_external_clients, monkeypatch, bs, tmp_path):
    single = FakeEmbedder()
    install(strict_external_clients, single.embed_content)
    ingest.build_index()
    ref_chunks, ref_vecs = published()
    n = len(ref_chunks)
    # fresh corpus copy, batched
    import shutil, os
    other = tmp_path / "b"
    shutil.copytree(corpus, other / "data" / "pdfs")
    os.chdir(other)
    batched = FakeEmbedder()
    install(strict_external_clients, batched.embed_content)
    monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", str(bs))
    stats = ingest.build_index()
    chunks, vecs = published()
    assert chunks == ref_chunks and np.array_equal(vecs, ref_vecs)
    assert single.count == n and batched.count == math.ceil(n / bs) == stats["embedding_requests"]
    assert stats["embeddings_computed"] == n and batched.items == n
    assert all(len(c["contents"]) <= bs for c in batched.calls)


def test_cache_misses_deduplicated_and_order_preserved(isolated_workdir, strict_external_clients, monkeypatch):
    folder = isolated_workdir / "data" / "pdfs"
    write_pdf(folder / "a.pdf", [golden_text("legal_service_agreement")])
    fake = FakeEmbedder()
    install(strict_external_clients, fake.embed_content)
    ingest.build_index()  # cache now holds the legal chunk
    for name, stem in (("b.pdf", "medical_blood_report"), ("c.pdf", "legal_service_agreement"), ("d.pdf", "medical_blood_report")):
        write_pdf(folder / name, [golden_text(stem)])
    monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", "8")
    stats = ingest.build_index()
    last = fake.calls[-1]["contents"]
    assert isinstance(last, list) and len(last) == 1 and "Haemoglobin" in last[0]  # one deduplicated miss
    assert stats["embeddings_computed"] == 1 and stats["cache_hits"] == 3
    chunks, vecs = published()
    assert [s for s, _ in chunks] == ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    for i, (_, t) in enumerate(chunks):
        assert np.array_equal(vecs[i], embed_text(t))


@pytest.mark.parametrize("bad", ["count", "nonfinite", "ragged"])
def test_invalid_response_publishes_nothing(corpus, strict_external_clients, monkeypatch, bad):
    fake = FakeEmbedder()
    install(strict_external_clients, fake.embed_content)
    write_pdf(corpus / "only.pdf", [golden_text("legal_service_agreement")])
    for s in STEMS + ["long"]:
        (corpus / f"{s}.pdf").unlink()
    ingest.build_index()
    before = index_store.signature()
    for s in STEMS:
        write_pdf(corpus / f"{s}.pdf", [golden_text(s)])

    def broken(*, model, contents):
        r = fake.embed_content(model=model, contents=contents)
        if bad == "count":
            r.embeddings = r.embeddings[:-1]
        elif bad == "nonfinite":
            r.embeddings[0].values[0] = float("nan")
        else:
            r.embeddings[0].values = r.embeddings[0].values[:10]
        return r
    install(strict_external_clients, broken)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", "8")
    with pytest.raises(Exception):
        ingest.build_index()
    assert index_store.signature() == before  # previous generation untouched


def test_partial_batch_failure_keeps_earlier_batches_and_retry_embeds_rest(corpus, strict_external_clients, monkeypatch):
    fake = FakeEmbedder()
    calls = {"n": 0}

    def fail_second_batch(*, model, contents):
        calls["n"] += 1
        if calls["n"] in (2, 3, 4):  # batch 2 fails on its first try and both retries
            raise RuntimeError("simulated outage")
        return fake.embed_content(model=model, contents=contents)
    install(strict_external_clients, fail_second_batch)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", "4")
    with pytest.raises(RuntimeError):
        ingest.build_index()
    assert not (corpus.parent / "index").exists()
    install(strict_external_clients, fake.embed_content)
    stats = ingest.build_index()
    assert stats["embeddings_computed"] == 9 - 4  # the first batch of 4 was kept in the cache
    chunks, vecs = published()
    assert all(np.array_equal(vecs[i], embed_text(t)) for i, (_, t) in enumerate(chunks))


def test_transient_failure_is_retried_and_counted(corpus, strict_external_clients, monkeypatch):
    fake = FakeEmbedder()
    calls = {"n": 0}

    def flaky(*, model, contents):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return fake.embed_content(model=model, contents=contents)
    install(strict_external_clients, flaky)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setenv("DOCINTEL_EMBED_BATCH_SIZE", "32")
    stats = ingest.build_index()
    assert stats["embedding_requests"] == 2 and stats["embeddings_computed"] == 9
