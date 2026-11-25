#!/usr/bin/env python3
"""P1B benchmark: original (load index + metadata per query) vs version-aware cached retrieval.

Local retrieval only (query embedding is a precomputed vector returned by a mock; no network,
no LLM). Both implementations read the SAME index files and receive the SAME query vectors.
Measured blocks alternate order across rounds. Raw samples are stored.

usage: python scripts/phase1_bench_retrieval.py [--large] [--out docs/phase1/bench/p1b_retrieval.json]
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bench_common  # noqa: F401
from bench_common import BASELINE_COMMIT, ROOT, baseline_source_sha, load_baseline
from bench_env import capture_environment, summarize_ms

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend import index_store, retriever  # noqa: E402

baseline_retriever = load_baseline("retriever")

CONFIGS = [(100, 768), (1000, 768), (5000, 768), (100, 3072), (1000, 3072), (5000, 3072)]
LARGE = [(20000, 768), (20000, 3072)]
N_QUERIES = 50
TOP_K = 4

_real_read_index = faiss.read_index
LOADS = {"n": 0}


def _counting_read_index(*a, **k):
    LOADS["n"] += 1
    return _real_read_index(*a, **k)


faiss.read_index = _counting_read_index  # both implementations call faiss.read_index


def make_dataset(root: Path, n: int, d: int, seed: int = 0) -> np.ndarray:
    """Publish an n x d corpus with the Phase 1 writer (standard faiss.index + chunks.pkl + manifest)."""
    rng = np.random.default_rng(seed + n * 7 + d)
    vectors = rng.standard_normal((n, d)).astype("float32")
    base = " ".join(f"word{j % 997}" for j in range(450))  # ~3.9 kB of text per chunk
    chunks = [(f"doc{i % 50:02d}.pdf", f"chunk {i} {base}") for i in range(n)]
    cwd = os.getcwd()
    os.chdir(root)
    try:
        index_store.publish(chunks, vectors, [{"source": "synthetic", "n_chunks": n}],
                            {"model": "synthetic", "dim": d}, f"synthetic-{n}-{d}")
    finally:
        os.chdir(cwd)
    return rng.standard_normal((N_QUERIES, d)).astype("float32")


def install_queries(qv: np.ndarray):
    values = [row.tolist() for row in qv]

    def embed(model, contents):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values[int(contents[1:])])])
    for mod in (baseline_retriever, retriever):
        mod.client = mock.Mock()
        mod.client.models.embed_content.side_effect = embed


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return (time.perf_counter() - t) * 1000, out


def republish_same_data():
    """New generation, identical content (as an upload of unchanged-content data would be if not skipped)."""
    gen = index_store.load_generation()
    vecs = gen.index.reconstruct_n(0, gen.index.ntotal)
    index_store.publish(gen.chunks, vecs, gen.manifest["documents"], gen.manifest["embedding"],
                        gen.manifest["corpus_fingerprint"] + "-r")


def tie_report(results_dir_qv):
    index = _real_read_index(index_store.INDEX_PATH)
    D, _ = index.search(results_dir_qv, TOP_K)
    ties = int(sum(np.any(np.isclose(np.diff(row), 0.0, atol=1e-6)) for row in D))
    return {"queries_with_tied_distances_in_top_k": ties}


def bench_config(n: int, d: int, rounds: int, large: bool) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"docintel-p1b-{n}-{d}-"))
    qv = make_dataset(root, n, d)
    install_queries(qv)
    cwd = os.getcwd()
    os.chdir(root)
    B = 10 if large else 25          # baseline queries per round (each one loads from disk)
    W = 100                          # optimized warm queries per round
    C = 8 if large else 15           # cold-cache repetitions
    I = 5 if large else 10           # first-query-after-invalidation repetitions
    raw = {"baseline": [], "optimized_warm": [], "optimized_cold": [], "optimized_after_invalidation": [],
           "baseline_load_only": [], "optimized_load_only": [], "faiss_search_only": []}
    loads = {}
    try:
        retriever.clear_cache()
        for j in range(3):  # warm-up (page cache, imports, allocator) for both, untimed
            baseline_retriever.retrieve(f"q{j}", TOP_K)
            retriever.retrieve(f"q{j}", TOP_K)
        q = 0
        for r in range(rounds):
            order = ("baseline", "optimized") if r % 2 == 0 else ("optimized", "baseline")
            for impl in order:
                if impl == "baseline":
                    before = LOADS["n"]
                    for _ in range(B):
                        raw["baseline"].append(timed(baseline_retriever.retrieve, f"q{q % N_QUERIES}", TOP_K)[0])
                        q += 1
                    loads.setdefault("baseline_per_query", []).append((LOADS["n"] - before) / B)
                else:
                    retriever.retrieve("q0", TOP_K)  # ensure warm (not timed)
                    before = LOADS["n"]
                    for _ in range(W):
                        raw["optimized_warm"].append(timed(retriever.retrieve, f"q{q % N_QUERIES}", TOP_K)[0])
                        q += 1
                    loads.setdefault("optimized_warm_total", []).append(LOADS["n"] - before)
        before = LOADS["n"]
        for c in range(C):
            retriever.clear_cache()
            raw["optimized_cold"].append(timed(retriever.retrieve, f"q{c % N_QUERIES}", TOP_K)[0])
        loads["optimized_cold_per_query"] = (LOADS["n"] - before) / C
        before = LOADS["n"]
        for c in range(I):
            retriever.retrieve("q0", TOP_K)  # warm on current generation
            republish_same_data()           # untimed: new generation published
            raw["optimized_after_invalidation"].append(timed(retriever.retrieve, f"q{c % N_QUERIES}", TOP_K)[0])
        loads["optimized_after_invalidation_per_query"] = (LOADS["n"] - before - I) / I  # minus warm-up loads
        for _ in range(C):
            def base_load():
                _real_read_index(baseline_retriever.INDEX_PATH)
                with open(baseline_retriever.CHUNKS_PATH, "rb") as f:
                    pickle.load(f)
            raw["baseline_load_only"].append(timed(base_load)[0])
            raw["optimized_load_only"].append(timed(index_store.load_generation)[0])
        gen = retriever.current_generation()
        for j in range(W):
            raw["faiss_search_only"].append(timed(gen.index.search, qv[j % N_QUERIES:j % N_QUERIES + 1], TOP_K)[0])

        # correctness equivalence on identical queries and embeddings
        retriever.clear_cache()
        eq = {"queries": N_QUERIES, "baseline_vs_optimized_warm_identical": 0, "baseline_vs_optimized_cold_identical": 0}
        for j in range(N_QUERIES):
            b = baseline_retriever.retrieve(f"q{j}", TOP_K)
            eq["baseline_vs_optimized_warm_identical"] += b == retriever.retrieve(f"q{j}", TOP_K)
            if j < 10:
                retriever.clear_cache()
                eq["baseline_vs_optimized_cold_identical"] += b == retriever.retrieve(f"q{j}", TOP_K)
        eq["cold_checked"] = 10
        eq.update(tie_report(qv))
        info = retriever.cache_info()
        resident = [e for e in info["entries"] if e["directory"] == os.path.realpath("data/index")][0]["resident_bytes"]
        sizes = {"index_file_bytes": os.path.getsize(index_store.INDEX_PATH),
                 "chunks_file_bytes": os.path.getsize(index_store.CHUNKS_PATH),
                 "manifest_bytes": os.path.getsize(index_store.MANIFEST_PATH),
                 "cached_generation_resident_bytes_estimate": resident, "vector_bytes": n * d * 4}
    finally:
        os.chdir(cwd)
    memory = {impl: mem_probe(impl, root) for impl in ("baseline", "optimized", "optimized_reload")}
    shutil.rmtree(root, ignore_errors=True)
    return {"n": n, "d": d, "raw_ms": raw, "summary": {k: summarize_ms(v) for k, v in raw.items()},
            "disk_loads": loads, "equivalence": eq, "sizes": sizes, "memory": memory,
            "plan": {"rounds": rounds, "baseline_queries_per_round": B, "optimized_warm_queries_per_round": W,
                     "cold_reps": C, "invalidation_reps": I, "warmup_queries_per_impl": 3}}


# ----------------------------------------------------------------------------- memory probe
def _status_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def _rss_kb():
    return _status_kb("VmRSS")


def _reset_peak():
    """Reset VmHWM to current RSS (Linux >= 4.0). ru_maxrss is NOT usable here: Linux carries
    it over from the parent across fork+exec."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def mem_probe(impl, root) -> dict:
    """Fresh interpreter per measurement: peak RSS growth while serving 20 queries."""
    out = subprocess.run([sys.executable, __file__, "--mem-probe", impl, str(root)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def run_mem_probe(impl, root):
    os.chdir(root)
    d = json.loads(Path(index_store.MANIFEST_PATH).read_text())["dim"]  # no index load before the snapshot
    qv = np.random.default_rng(1).standard_normal((N_QUERIES, d)).astype("float32")
    install_queries(qv)
    reset = _reset_peak()
    rss0 = _rss_kb()
    fn = baseline_retriever.retrieve if impl == "baseline" else retriever.retrieve
    for j in range(20):
        fn(f"q{j}", TOP_K)
        if impl == "optimized_reload" and j == 9:
            # new generation published by ANOTHER process (writer memory excluded); the reader's
            # next query reloads while its previous generation is still referenced
            subprocess.run([sys.executable, __file__, "--republish", str(root)], check=True, capture_output=True)
    hwm = _status_kb("VmHWM")
    print(json.dumps({"rss_before_kb": rss0, "rss_after_kb": _rss_kb(), "peak_rss_kb": hwm,
                      "peak_rss_growth_kb": hwm - rss0 if reset else None,
                      "retained_rss_growth_kb": _rss_kb() - rss0, "peak_reset_supported": reset}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--mem-probe":
        run_mem_probe(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--republish":
        os.chdir(sys.argv[2])
        republish_same_data()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs/phase1/bench/p1b_retrieval.json"))
    args = ap.parse_args()
    result = {"benchmark": "P1B local retrieval", "baseline_commit": BASELINE_COMMIT,
              "baseline_retriever_blob": baseline_source_sha("retriever"), "environment": capture_environment(),
              "methodology": {
                  "scope": "retrieve() wall time: index/metadata availability + query-vector conversion + FAISS search"
                           " + result mapping. Query embedding is a precomputed vector (mock); no network or LLM time.",
                  "inputs": f"{N_QUERIES} seeded random query vectors per config, identical for both implementations;"
                            " corpus vectors seeded standard normal; ~3.9 kB text per chunk",
                  "order": "rounds alternate which implementation is measured first",
                  "warmup": "3 untimed queries per implementation per config (OS page cache warm for both)",
                  "percentiles": "nearest-rank on raw samples", "top_k": TOP_K,
                  "memory": "separate fresh interpreter per implementation; VmHWM reset via /proc/self/clear_refs after imports, then peak and retained RSS growth over 20 queries"},
              "configs": []}
    for n, d in CONFIGS + (LARGE if args.large else []):
        res = bench_config(n, d, args.rounds, large=n > 5000)
        result["configs"].append(res)
        s = res["summary"]
        print(f"n={n:>6} d={d:>4}  baseline p50 {s['baseline']['p50_ms']:8.3f}  warm p50 {s['optimized_warm']['p50_ms']:7.3f}"
              f"  cold p50 {s['optimized_cold']['p50_ms']:8.3f}  inval p50 {s['optimized_after_invalidation']['p50_ms']:8.3f}"
              f"  search {s['faiss_search_only']['p50_ms']:6.3f}  eq {res['equivalence']['baseline_vs_optimized_warm_identical']}/50"
              f"  mem peak KB b={res['memory']['baseline']['peak_rss_growth_kb']} o={res['memory']['optimized']['peak_rss_growth_kb']}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
