#!/usr/bin/env python3
"""SIMULATED-PROVIDER application load test: baseline main.py (d27da24) vs current main.py.

Both apps serve /chat from the same prebuilt index with the same simulated blocking provider:
query embedding sleeps EMBED_MS, LLM generation sleeps GEN_MS (time.sleep, like the synchronous
SDK calls). In-process ASGI transport (httpx.ASGITransport): client and server share one event
loop, so a blocked loop delays everything, which is exactly what is being measured.
NOT live production latency; no network, no real provider.
"""
import asyncio
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["GEMINI_API_KEY"] = os.environ["GROQ_API_KEY"] = "offline-not-used"
for t in ("google.genai.Client", "groq.Groq"):
    mock.patch(t).start()
import httpx  # noqa: E402
from bench_env import capture_environment, summarize_ms  # noqa: E402
from backend import ingest, llm, main, retriever  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402
from tests.pdfutil import golden_text, write_pdf  # noqa: E402

BASELINE_COMMIT = "d27da24784211ad6fff8e017775676546fb911d6"
EMBED_MS, GEN_MS = 30, 300
LEVELS, REQUESTS_PER_LEVEL, HEALTH_EVERY_S = [1, 5, 10, 20], 40, 0.025


class Provider:
    def __init__(self):
        self.fake, self.lock, self.active, self.peak = FakeEmbedder(), threading.Lock(), 0, 0

    def _enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def _exit(self):
        with self.lock:
            self.active -= 1

    def embed(self, **kw):
        self._enter()
        time.sleep(EMBED_MS / 1000)
        self._exit()
        return self.fake.embed_content(**kw)

    def generate(self, **kw):
        self._enter()
        time.sleep(GEN_MS / 1000)
        self._exit()
        return mock.Mock(text="SIMULATED_ANSWER")


def load_baseline_app():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE_COMMIT}:backend/main.py"],
                         check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "baseline_main.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("baseline_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


async def level(app, provider, c):
    lat, health, errors = [], [], 0
    stop = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as cl:
        async def probe():
            while not stop.is_set():
                t = time.perf_counter()
                await cl.get("/health")
                health.append((time.perf_counter() - t) * 1000)
                await asyncio.sleep(HEALTH_EVERY_S)
        queue = list(range(REQUESTS_PER_LEVEL))

        async def worker():
            nonlocal errors
            while queue:
                i = queue.pop()
                t = time.perf_counter()
                r = await cl.post("/chat", json={"question": f"annual leave policy {i}"})
                lat.append((time.perf_counter() - t) * 1000)
                errors += r.status_code != 200 or r.json().get("answer") != "SIMULATED_ANSWER"
        p = asyncio.create_task(probe())
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(c)])
        wall = time.perf_counter() - t0
        stop.set()
        await p
    return {"concurrency": c, "requests": len(lat), "failures": errors, "wall_s": round(wall, 3),
            "throughput_rps": round(len(lat) / wall, 2), "latency": summarize_ms(lat), "latency_raw_ms": lat,
            "health": summarize_ms(health), "health_raw_ms": health, "provider_peak_concurrency": provider.peak}


def main_():
    work = Path(tempfile.mkdtemp(prefix="docintel-load-"))
    os.chdir(work)
    for s in ["enterprise_leave_policy", "legal_service_agreement", "medical_blood_report", "study_photosynthesis_notes"]:
        write_pdf(work / "data/pdfs" / f"{s}.pdf", [golden_text(s)])
    ingest.client = mock.Mock()
    ingest.client.models.embed_content.side_effect = FakeEmbedder().embed_content
    ingest.build_index()
    apps = {"baseline": load_baseline_app(), "optimized": main.app}
    out = {"benchmark": "SIMULATED-provider application load test", "baseline_commit": BASELINE_COMMIT,
           "simulated_provider": {"query_embedding_ms": EMBED_MS, "generation_ms": GEN_MS, "blocking": "time.sleep"},
           "optimized_bounds": {"max_concurrent_chat": main.MAX_CONCURRENT_CHAT},
           "workload": {"levels": LEVELS, "requests_per_level": REQUESTS_PER_LEVEL, "health_probe_interval_s": HEALTH_EVERY_S},
           "environment": capture_environment(), "results": {"baseline": [], "optimized": []}}
    for i, c in enumerate(LEVELS):
        for name in (("baseline", "optimized") if i % 2 == 0 else ("optimized", "baseline")):
            provider = Provider()
            retriever.client = mock.Mock()
            retriever.client.models.embed_content.side_effect = provider.embed
            llm.gemini_client = mock.Mock()
            llm.gemini_client.models.generate_content.side_effect = provider.generate
            retriever.clear_cache()
            if os.path.exists("data/audit_log.csv"):
                os.remove("data/audit_log.csv")
            res = asyncio.run(level(apps[name], provider, c))
            with open("data/audit_log.csv") as f:
                res["audit_rows"] = sum(1 for _ in __import__("csv").reader(f))
            out["results"][name].append(res)
            print(f"{name:9s} c={c:2d} rps {res['throughput_rps']:6.2f} p50 {res['latency']['p50_ms']:8.1f} "
                  f"p95 {res['latency']['p95_ms']:8.1f} fail {res['failures']} health p95 {res['health']['p95_ms']:7.1f} "
                  f"max {res['health']['max_ms']:7.1f} peak-provider {res['provider_peak_concurrency']} audit {res['audit_rows']}", flush=True)
    dest = ROOT / "docs" / "phase4" / "load_test.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_()
