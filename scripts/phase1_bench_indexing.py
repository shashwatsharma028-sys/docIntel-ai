#!/usr/bin/env python3
"""P1A benchmark: original vs incremental indexing on identical synthetic workloads.

Offline only. The embedding provider is the test-suite FakeEmbedder (zero network latency),
so invocation counts are exact but wall times EXCLUDE all provider/network latency.
Both implementations run in the same process; repetitions alternate execution order.

usage: python scripts/phase1_bench_indexing.py [--reps 3] [--out docs/phase1/bench/p1a_indexing.json]
"""
import argparse
import json
import os
import pickle
import random
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import bench_common  # noqa: F401  (patches SDKs, sets sys.path)
from bench_common import BASELINE_COMMIT, ROOT, baseline_source_sha, load_baseline, quiet
from bench_env import capture_environment, summarize_ms

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend import embedding_cache, index_store, ingest, retriever  # noqa: E402
from tests.fakes import FakeEmbedder, embed_text  # noqa: E402
from tests.pdfutil import make_pdf_bytes  # noqa: E402

baseline_ingest = load_baseline("ingest")

# ----------------------------------------------------------------------------- synthetic corpus
_rng = random.Random(1234)
VOCAB = sorted({"".join(_rng.choice("bcdfghjklmnprstvwz") + _rng.choice("aeiou") for _ in range(_rng.randint(2, 4)))
                for _ in range(4000)})


def doc_words(seed: int, n: int) -> list[str]:
    r = random.Random(seed)
    return [r.choice(VOCAB) for _ in range(n)]


_PDF_CACHE: dict = {}


def pdf_bytes(seed: int, n_words: int, mutate_tail: int = 0) -> bytes:
    key = (seed, n_words, mutate_tail)
    if key not in _PDF_CACHE:
        words = doc_words(seed, n_words)
        if mutate_tail:  # change words in the final 100 words only -> only the last chunk changes
            for j in range(n_words - mutate_tail, n_words, 7):
                words[j] = f"revised{j}"
        pages = [" ".join(words[i:i + 300]) for i in range(0, n_words, 300)]
        _PDF_CACHE[key] = make_pdf_bytes(pages)
    return _PDF_CACHE[key]


SINGLE = 150   # words -> 1 chunk
MULTI = 3000   # words -> 5 chunks (windows at 0, 700, 1400, 2100, 2800)


# ----------------------------------------------------------------------------- workloads
# Each workload: (setup_steps, measured_steps). A step is a list of folder actions
# ("put", name, bytes) / ("rm", name) followed by one build_index() call.
def w_sequential(n):
    return [], [[("put", f"doc{i:03d}.pdf", pdf_bytes(i, SINGLE))] for i in range(n)]


def w_multichunk(n=10):
    return [], [[("put", f"multi{i:02d}.pdf", pdf_bytes(1000 + i, MULTI))] for i in range(n)]


def w_identical_reupload(n=10, reps=10):
    setup = [[("put", f"doc{i:03d}.pdf", pdf_bytes(i, SINGLE)) for i in range(n)]]
    return setup, [[("put", f"doc{k % n:03d}.pdf", pdf_bytes(k % n, SINGLE))] for k in range(reps)]


def w_modified(n=10):
    setup = [[("put", f"multi{i:02d}.pdf", pdf_bytes(1000 + i, MULTI)) for i in range(n)]]
    return setup, [[("put", f"multi{i:02d}.pdf", pdf_bytes(1000 + i, MULTI, mutate_tail=100))] for i in range(n)]


def w_mixed(existing=50, new=10):
    setup = [[("put", f"doc{i:03d}.pdf", pdf_bytes(i, SINGLE)) for i in range(existing)]]
    batch = [("put", f"new{i:03d}.pdf", pdf_bytes(500 + i, SINGLE)) for i in range(new)]
    return setup, [batch + [("rm", "doc000.pdf"), ("put", "doc001.pdf", pdf_bytes(9001, SINGLE))]]


WORKLOADS = {
    "sequential_unique_single_chunk_10": lambda: w_sequential(10),
    "sequential_unique_single_chunk_50": lambda: w_sequential(50),
    "sequential_unique_single_chunk_100": lambda: w_sequential(100),
    "sequential_multi_chunk_10x5": lambda: w_multichunk(10),
    "identical_reupload_10_into_10": lambda: w_identical_reupload(10, 10),
    "modified_last_chunk_10_multi_chunk_docs": lambda: w_modified(10),
    "mixed_50_existing_plus_10_new_1_removed_1_replaced": lambda: w_mixed(50, 10),
}


# ----------------------------------------------------------------------------- instrumentation
class _TimedFaiss:
    """Times baseline index construction+write: IndexFlatL2() .. write_index() + chunk pickle."""
    def __init__(self, real):
        self._real, self.t0, self.ms = real, None, None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def IndexFlatL2(self, d):
        self.t0 = time.perf_counter()
        return self._real.IndexFlatL2(d)


class _TimedPickle:
    def __init__(self, real, tf):
        self._real, self._tf = real, tf

    def __getattr__(self, name):
        return getattr(self._real, name)

    def dump(self, obj, f, *a, **k):
        self._real.dump(obj, f, *a, **k)
        self._tf.ms = (time.perf_counter() - self._tf.t0) * 1000


_timed_faiss = _TimedFaiss(faiss)
baseline_ingest.faiss = _timed_faiss
baseline_ingest.pickle = _TimedPickle(pickle, _timed_faiss)


class Impl:
    def __init__(self, name):
        self.name = name
        self.fake = FakeEmbedder()
        self.fail_at = None  # fail on the k-th embedding call of the next build (1-based)
        self.module = baseline_ingest if name == "baseline" else ingest
        self.module.client = mock.Mock()
        self.module.client.models.embed_content.side_effect = self._embed

    def _embed(self, **kw):
        if self.fail_at is not None:
            self.fail_at -= 1
            if self.fail_at == 0:
                self.fail_at = None
                raise RuntimeError("simulated embedding outage")
        return self.fake.embed_content(**kw)

    def build(self):
        calls = self.fake.count
        _timed_faiss.ms = None
        t = time.perf_counter()
        error, stats = None, {}
        try:
            with quiet():
                stats = self.module.build_index() or {}
        except RuntimeError as e:
            error = str(e)
        ms = (time.perf_counter() - t) * 1000
        step = {"latency_ms": ms, "embed_calls": self.fake.count - calls, "error": error}
        if self.name == "baseline":
            step.update(embedded_items=step["embed_calls"], cache_hits=0, cache_misses=step["embed_calls"],
                        index_publish_ms=_timed_faiss.ms)
        else:
            step.update(embedded_items=stats.get("embeddings_computed", step["embed_calls"]),
                        cache_hits=stats.get("cache_hits", 0), cache_misses=stats.get("embeddings_computed", 0),
                        index_publish_ms=stats.get("publish_ms"), index_construct_ms=stats.get("index_construct_ms"),
                        documents_parsed=stats.get("documents_parsed"), documents_reused=stats.get("documents_reused"),
                        unchanged=stats.get("unchanged"))
        return step


def apply(folder: Path, actions):
    for act in actions:
        if act[0] == "put":
            (folder / act[1]).write_bytes(act[2])
        else:
            (folder / act[1]).unlink()


def dir_bytes(path: str) -> int:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def run_workload(impl_name: str, spec) -> dict:
    setup, measured = spec
    impl = Impl(impl_name)
    work = Path(tempfile.mkdtemp(prefix=f"docintel-p1a-{impl_name}-"))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        folder = work / "data" / "pdfs"
        folder.mkdir(parents=True)
        for actions in setup:
            apply(folder, actions)
            impl.build()
        setup_calls = impl.fake.count
        steps = []
        for actions in measured:
            apply(folder, actions)
            steps.append(impl.build())
        return {"setup_embed_calls": setup_calls, "steps": steps,
                "storage_bytes": {"index_dir": dir_bytes("data/index"), "embedding_cache": dir_bytes("data/embedding_cache")},
                "workdir": str(work)}
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------------- failure workload
def run_failure(impl_name: str) -> dict:
    """10 committed docs; add 5 new; the 3rd embedding call of that build fails; then retry."""
    impl = Impl(impl_name)
    work = Path(tempfile.mkdtemp(prefix=f"docintel-p1a-fail-{impl_name}-"))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        folder = work / "data" / "pdfs"
        folder.mkdir(parents=True)
        apply(folder, [("put", f"doc{i:03d}.pdf", pdf_bytes(i, SINGLE)) for i in range(10)])
        impl.build()
        committed = {p: Path(p).read_bytes() for p in ("data/index/faiss.index", "data/index/chunks.pkl")}
        apply(folder, [("put", f"new{i:03d}.pdf", pdf_bytes(700 + i, SINGLE)) for i in range(5)])
        impl.fail_at = 3
        failed = impl.build()
        intact = all(Path(p).read_bytes() == b for p, b in committed.items())
        retry = impl.build()
        return {"failed_build": failed, "previous_index_byte_identical_after_failure": intact, "retry_build": retry}
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------------- equivalence
EQ_QUERIES = [" ".join(doc_words(90000 + q, 6)) for q in range(20)] + \
             [" ".join(doc_words(i, SINGLE)[:20]) for i in range(10)]


def _load_pair(workdir):
    index = faiss.read_index(str(Path(workdir) / "data/index/faiss.index"))
    with open(Path(workdir) / "data/index/chunks.pkl", "rb") as f:
        chunks = index_store.loads_chunks(f.read())
    return index, chunks


def _ranked(index, chunks, q, k):
    D, I = index.search(np.array([embed_text(q)], dtype="float32"), k)
    return [(round(float(d), 5), chunks[i]) for d, i in zip(D[0], I[0]) if 0 <= i < len(chunks)]


def compare(workdir_a, workdir_b, k=4) -> dict:
    ia, ca = _load_pair(workdir_a)
    ib, cb = _load_pair(workdir_b)
    out = {"same_chunk_multiset": sorted(ca) == sorted(cb), "queries": len(EQ_QUERIES),
           "exact": 0, "equal_up_to_ties": 0, "different": 0}
    for q in EQ_QUERIES:
        ra, rb = _ranked(ia, ca, q, k), _ranked(ib, cb, q, k)
        if ra == rb:
            out["exact"] += 1
        else:
            # equal distances may be ordered differently (e.g. different chunk order on disk);
            # compare complete tie groups only (the last group may be cut by k).
            ga = {d: sorted(x for dd, x in ra if dd == d) for d, _ in ra}
            gb = {d: sorted(x for dd, x in rb if dd == d) for d, _ in rb}
            dists = sorted(set(ga) | set(gb))
            last = dists[-1] if dists else None
            same = [d for d, _ in ra] == [d for d, _ in rb] and all(ga.get(d) == gb.get(d) for d in dists if d != last)
            out["equal_up_to_ties" if same else "different"] += 1
    return out


def clean_rebuild(workdir) -> str:
    """Fresh optimized build of the same final folder with no cache and no previous generation."""
    clean = Path(tempfile.mkdtemp(prefix="docintel-p1a-clean-"))
    shutil.copytree(Path(workdir) / "data" / "pdfs", clean / "data" / "pdfs")
    impl = Impl("optimized")
    cwd = os.getcwd()
    os.chdir(clean)
    try:
        impl.build()
    finally:
        os.chdir(cwd)
    return str(clean)


def optimized_vs_clean_exact(workdir, clean) -> dict:
    ia, ca = _load_pair(workdir)
    ib, cb = _load_pair(clean)
    return {"identical_chunk_lists": ca == cb,
            "identical_vectors": bool(np.array_equal(ia.reconstruct_n(0, ia.ntotal), ib.reconstruct_n(0, ib.ntotal))),
            **{k: v for k, v in compare(workdir, clean).items() if k != "same_chunk_multiset"}}


# ----------------------------------------------------------------------------- main
def summarize(runs):
    lat = [s["latency_ms"] for r in runs for s in r["steps"]]
    first = runs[0]
    return {
        "measured_embed_calls": sum(s["embed_calls"] for s in first["steps"]),
        "embed_calls_per_step": [s["embed_calls"] for s in first["steps"]],
        "embedded_items": sum(s["embedded_items"] for s in first["steps"]),
        "cache_hits": sum(s["cache_hits"] for s in first["steps"]),
        "cache_misses": sum(s["cache_misses"] for s in first["steps"]),
        "setup_embed_calls": first["setup_embed_calls"],
        "per_step_latency": summarize_ms(lat),
        "cumulative_latency_ms_per_rep": [round(sum(s["latency_ms"] for s in r["steps"]), 3) for r in runs],
        "index_publish": summarize_ms([s["index_publish_ms"] for r in runs for s in r["steps"] if s["index_publish_ms"]])
        if any(s["index_publish_ms"] for r in runs for s in r["steps"]) else None,
        "storage_bytes": first["storage_bytes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs/phase1/bench/p1a_indexing.json"))
    ap.add_argument("--only")
    args = ap.parse_args()
    env = capture_environment()
    result = {"benchmark": "P1A indexing", "baseline_commit": BASELINE_COMMIT,
              "baseline_ingest_blob": baseline_source_sha("ingest"), "environment": env,
              "methodology": {
                  "provider": "tests.fakes.FakeEmbedder (offline, 256-d, zero latency)",
                  "repetitions": args.reps, "order": "alternating per repetition (even: baseline first)",
                  "warmup": "one untimed build of an unrelated 3-doc corpus per implementation before measuring",
                  "latency_scope": "build_index() wall time excluding all network/provider latency",
                  "index_publish_ms": "baseline: IndexFlatL2()..faiss.write_index+pickle.dump; optimized: publish()"
                                      " = construct+serialize+write index/chunks/manifest+atomic swap"},
              "workloads": {}}
    for impl in ("baseline", "optimized"):  # warm-up
        run_workload(impl, ([], [[("put", f"warm{i}.pdf", pdf_bytes(77000 + i, SINGLE))] for i in range(3)]))
    for name, factory in WORKLOADS.items():
        if args.only and args.only not in name:
            continue
        runs = {"baseline": [], "optimized": []}
        for rep in range(args.reps):
            order = ("baseline", "optimized") if rep % 2 == 0 else ("optimized", "baseline")
            for impl in order:
                runs[impl].append(run_workload(impl, factory()))
        eq = {"optimized_incremental_vs_clean_optimized_rebuild":
              optimized_vs_clean_exact(runs["optimized"][0]["workdir"], clean_rebuild(runs["optimized"][0]["workdir"])),
              "optimized_vs_baseline_final_index": compare(runs["optimized"][0]["workdir"], runs["baseline"][0]["workdir"])}
        result["workloads"][name] = {"raw": runs, "summary": {k: summarize(v) for k, v in runs.items()},
                                     "retrieval_equivalence": eq}
        s = result["workloads"][name]["summary"]
        print(f"{name}: embed calls baseline={s['baseline']['measured_embed_calls']} "
              f"optimized={s['optimized']['measured_embed_calls']}  "
              f"p50 step ms {s['baseline']['per_step_latency']['p50_ms']} -> {s['optimized']['per_step_latency']['p50_ms']}  "
              f"eq={eq['optimized_incremental_vs_clean_optimized_rebuild']['identical_vectors']}/"
              f"{eq['optimized_vs_baseline_final_index']['different']} diff")
    if not args.only:
        result["failure_workload"] = {impl: run_failure(impl) for impl in ("baseline", "optimized")}
        print("failure:", json.dumps({k: {"failed_calls": v["failed_build"]["embed_calls"],
                                          "retry_calls": v["retry_build"]["embed_calls"],
                                          "intact": v["previous_index_byte_identical_after_failure"]}
                                      for k, v in result["failure_workload"].items()}))
    for wl in result["workloads"].values():
        for runs in wl["raw"].values():
            for r in runs:
                shutil.rmtree(r.pop("workdir"), ignore_errors=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
