#!/usr/bin/env python3
"""Embedding-batching benchmark through the real build_index(), SIMULATED provider.

Corpus: the 49 frozen QASPER papers (11 dev + 38 test), dataset text rendered to PDFs
(reconstructed PDFs, not original layouts), default 800/100-word chunker, cold cache.
Provider: deterministic offline embedder wrapped in a latency MODEL (assumption, not measured):
per request BASE_MS + PER_ITEM_MS x items. Request and item counts are exact; wall times depend
on the assumed model (two models reported). Same items embedded in every configuration, so
provider tokens/billing are unchanged by batching; only the number of HTTP requests changes.
"""
import json
import math
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation" / "qasper"))
os.environ["GEMINI_API_KEY"] = os.environ["GROQ_API_KEY"] = "offline-not-used"
for t in ("google.genai.Client", "groq.Groq"):
    mock.patch(t).start()
import faiss  # noqa: E402
import numpy as np  # noqa: E402
from bench_env import capture_environment, summarize_ms  # noqa: E402
from backend import ingest, retriever  # noqa: E402
from tests.fakes import FakeEmbedder, embed_text  # noqa: E402
from tests.pdfutil import make_pdf_bytes  # noqa: E402

BATCH_SIZES, REPS = [1, 8, 16, 32], 3
LATENCY_MODELS = {"model_A": (120, 4), "model_B": (60, 1)}  # (base ms per request, ms per item) -- assumptions


def corpus_pdfs():
    import run_qasper_eval as rq
    frozen = json.loads(rq.FROZEN.read_text())
    rows = {r["id"]: r for split in ("validation", "test") for r in rq.load_split(split)}
    out = {}
    for part in ("dev", "test"):
        for p in frozen[part]["papers"]:
            ft = rows[p["paper_id"]]["full_text"]
            words = " ".join(" ".join(ps) for ps in ft["paragraphs"]).split()
            pages = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
            out[f"{p['paper_id']}.pdf"] = make_pdf_bytes(pages)
    return out


def run(pdfs, bs, latency):
    work = Path(tempfile.mkdtemp(prefix="docintel-batch-"))
    for name, b in pdfs.items():
        (work / "data/pdfs").mkdir(parents=True, exist_ok=True)
        (work / "data/pdfs" / name).write_bytes(b)
    fake = FakeEmbedder()
    base_ms, item_ms = latency

    def provider(**kw):
        n = len(kw["contents"]) if isinstance(kw["contents"], list) else 1
        time.sleep((base_ms + item_ms * n) / 1000)
        return fake.embed_content(**kw)
    ingest.client = mock.Mock()
    ingest.client.models.embed_content.side_effect = provider
    cwd = os.getcwd()
    os.chdir(work)
    os.environ["DOCINTEL_EMBED_BATCH_SIZE"] = str(bs)
    tracemalloc.start()
    t = time.perf_counter()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        stats = ingest.build_index()
    wall = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    idx = faiss.read_index(ingest.INDEX_PATH)
    import pickle
    with open(ingest.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    vecs = idx.reconstruct_n(0, idx.ntotal)
    os.chdir(cwd)
    shutil.rmtree(work, ignore_errors=True)
    return {"http_requests": fake.count, "items": fake.items, "stats_requests": stats.get("embedding_requests"),
            "wall_s": wall, "items_per_s": fake.items / wall, "python_peak_mb": peak / 1e6,
            "chunks": chunks, "vecs": vecs}


def main():
    pdfs = corpus_pdfs()
    out = {"benchmark": "embedding batching, SIMULATED provider latency", "papers": len(pdfs),
           "latency_models_ms": LATENCY_MODELS, "reps": REPS, "environment": capture_environment(), "results": {}}
    for mname, lat in LATENCY_MODELS.items():
        ref = None
        for bs in BATCH_SIZES:
            runs = [run(pdfs, bs, lat) for _ in range(REPS)]
            r0 = runs[0]
            if ref is None:
                ref = r0
            eq = r0["chunks"] == ref["chunks"] and bool(np.array_equal(r0["vecs"], ref["vecs"]))
            # top-4 equivalence on 30 queries built from chunk text
            index = faiss.IndexFlatL2(r0["vecs"].shape[1]); index.add(r0["vecs"])
            rindex = faiss.IndexFlatL2(ref["vecs"].shape[1]); rindex.add(ref["vecs"])
            qs = np.stack([embed_text(" ".join(c[1].split()[:12])) for c in ref["chunks"][::max(1, len(ref["chunks"]) // 30)][:30]])
            top_eq = bool((index.search(qs, 4)[1] == rindex.search(qs, 4)[1]).all())
            walls = [r["wall_s"] * 1000 for r in runs]
            res = {"batch_size": bs, "http_requests": r0["http_requests"], "embedded_items": r0["items"],
                   "requests_reported_by_build_stats": r0["stats_requests"],
                   "wall": summarize_ms(walls), "wall_raw_ms": walls,
                   "items_per_s_mean": round(sum(r["items_per_s"] for r in runs) / REPS, 1),
                   "python_peak_mb_max": round(max(r["python_peak_mb"] for r in runs), 2),
                   "identical_to_batch1": eq, "top4_identical_to_batch1": top_eq}
            out["results"].setdefault(mname, []).append(res)
            print(mname, json.dumps({k: v for k, v in res.items() if k != "wall_raw_ms"}), flush=True)
    dest = ROOT / "docs" / "phase4" / "batch_bench.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
