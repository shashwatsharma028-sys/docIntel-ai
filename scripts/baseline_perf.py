#!/usr/bin/env python3
"""Offline baseline performance capture (Phase 0). No network, no credentials.

Measures what CAN be measured offline at the baseline checkpoint:
  * PDF extraction and chunking throughput (real PyMuPDF).
  * Embedding API call counts per upload (the dominant real-world cost; D-03),
    using a zero-latency fake embedder -- wall times here EXCLUDE network latency.
  * /chat retrieval latency vs index size, split into disk load (index + pickle,
    repeated on every query) and FAISS search, with random vectors at d=768 and d=3072.
Live Gemini/Groq latency is deliberately NOT measured (live API restriction).

usage: python scripts/baseline_perf.py [--out docs/baseline_performance.json]
"""
import argparse
import json
import os
import pickle
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["GEMINI_API_KEY"] = os.environ["GROQ_API_KEY"] = "offline-perf-not-real"
for target in ("google.genai.Client", "groq.Groq"):
    mock.patch(target).start()

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend import ingest, retriever  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402
from tests.pdfutil import GOLDEN_DOCS, golden_text, make_pdf_bytes, numbered_words  # noqa: E402


def timed(fn, repeat):
    xs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": round(statistics.mean(xs), 3), "p95_ms": round(sorted(xs)[int(0.95 * (len(xs) - 1))], 3),
            "min_ms": round(min(xs), 3), "runs": repeat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "baseline_performance.json"))
    args = ap.parse_args()
    res = {"checkpoint": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "python": platform.python_version(), "machine": platform.machine(),
           "faiss": faiss.__version__, "numpy": np.__version__,
           "note": "Offline, mocked embeddings/LLM. Wall times exclude all network/API latency."}
    fake = FakeEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        ingest.client = mock.Mock()
        retriever.client = mock.Mock()
        ingest.client.models.embed_content.side_effect = fake.embed_content
        retriever.client.models.embed_content.side_effect = fake.embed_content

        # 1. extraction + chunking: 20 pages x 400 words
        pdf = Path(tmp) / "big.pdf"
        pdf.write_bytes(make_pdf_bytes([numbered_words(400, prefix=f"p{p}w") for p in range(20)]))
        res["parse_pdf_20_pages_8000_words"] = timed(lambda: ingest.parse_pdf(str(pdf)), 20)
        text = numbered_words(100_000)
        res["chunk_text_100k_words"] = timed(lambda: ingest.chunk_text(text), 20)
        res["chunk_text_100k_words"]["chunks"] = len(ingest.chunk_text(text))

        # 2. sequential uploads: embedding calls grow with corpus size (full re-index each time)
        os.makedirs("data/pdfs", exist_ok=True)
        docs = [golden_text(s) for s in GOLDEN_DOCS]
        per_upload, cumulative, walls = [], [], []
        import builtins
        real_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            for n in range(1, 11):
                Path(f"data/pdfs/doc{n:02d}.pdf").write_bytes(make_pdf_bytes([docs[n % 4] + f" doc{n}"]))
                before = fake.count
                t = time.perf_counter()
                ingest.build_index()
                walls.append(round((time.perf_counter() - t) * 1000, 2))
                per_upload.append(fake.count - before)
                cumulative.append(fake.count)
        finally:
            builtins.print = real_print
        res["sequential_uploads_10_single_chunk_docs"] = {
            "embedding_calls_per_upload": per_upload, "cumulative_embedding_calls": cumulative,
            "build_index_wall_ms_zero_latency": walls,
            "ideal_incremental_cumulative_calls": list(range(1, 11)),
            "projection": "real cost ~= calls x per-call Gemini latency; cumulative grows O(n^2) in uploads"}

        # 3. retrieval latency vs index size (disk load happens on EVERY query)
        rng = np.random.default_rng(0)
        retrieval = {}
        for dim in (768, 3072):
            for n in (100, 1000, 5000):
                vecs = rng.standard_normal((n, dim)).astype("float32")
                idx = faiss.IndexFlatL2(dim)
                idx.add(vecs)
                faiss.write_index(idx, retriever.INDEX_PATH)
                with open(retriever.CHUNKS_PATH, "wb") as f:
                    pickle.dump([(f"d{i % 50}.pdf", "lorem ipsum " * 300) for i in range(n)], f)
                qv = rng.standard_normal(dim).astype("float32").tolist()
                retriever.client.models.embed_content.side_effect = None
                retriever.client.models.embed_content.return_value = mock.Mock(embeddings=[mock.Mock(values=qv)])

                def load():
                    faiss.read_index(retriever.INDEX_PATH)
                    with open(retriever.CHUNKS_PATH, "rb") as f:
                        pickle.load(f)

                q = np.array([qv], dtype="float32")
                retrieval[f"d{dim}_n{n}"] = {
                    "retrieve_total": timed(lambda: retriever.retrieve("q"), 30),
                    "disk_load_only": timed(load, 30),
                    "faiss_search_only": timed(lambda: idx.search(q, 4), 30),
                    "index_bytes": os.path.getsize(retriever.INDEX_PATH),
                }
        res["retrieve_latency"] = retrieval
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
