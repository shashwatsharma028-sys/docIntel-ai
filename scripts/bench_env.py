"""Benchmark environment capture and statistics helpers (Phase 1 methodology).

Every Phase 1 benchmark result file embeds `capture_environment()` so that numbers
are only ever compared within one recorded environment.
"""
import importlib.metadata as md
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_PACKAGES = ["faiss-cpu", "numpy", "pymupdf", "google-genai", "groq", "fastapi", "starlette",
                "python-multipart", "python-dotenv", "httpx", "pytest"]


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def capture_environment() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    model = next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("model name")), platform.processor())
    flags = next((l.split(":", 1)[1].split() for l in cpuinfo.splitlines() if l.startswith("flags")), [])
    meminfo = {l.split(":")[0]: l.split(":")[1].strip() for l in _read("/proc/meminfo").splitlines() if ":" in l}
    try:
        import faiss
        faiss_threads = faiss.omp_get_max_threads()
    except Exception:  # pragma: no cover
        faiss_threads = None
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "backend", "scripts", "tests")),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "machine": platform.machine(),
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "sched_affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_simd_flags": sorted(f for f in flags if f in {"avx", "avx2", "avx512f", "fma", "sse4_2", "sha_ni"}),
        "mem_total": meminfo.get("MemTotal"),
        "mem_available_at_start": meminfo.get("MemAvailable"),
        "threads": {"faiss_omp_max_threads": faiss_threads,
                    **{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "packages_key": {p: _version(p) for p in KEY_PACKAGES},
        "packages_all": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()),
    }


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def summarize_ms(samples: list[float]) -> dict:
    """p50/p95 use nearest-rank on the sorted raw samples; raw samples are stored separately."""
    xs = sorted(samples)
    n = len(xs)
    rank = lambda q: xs[min(n - 1, max(0, math.ceil(q * n) - 1))]  # nearest-rank
    return {"n": n, "mean_ms": round(statistics.fmean(xs), 4), "p50_ms": round(rank(0.50), 4),
            "p95_ms": round(rank(0.95), 4), "min_ms": round(xs[0], 4), "max_ms": round(xs[-1], 4),
            "stdev_ms": round(statistics.stdev(xs), 4) if n > 1 else 0.0}
