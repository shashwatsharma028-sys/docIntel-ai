# DocIntel AI — Multi-Domain RAG Document Q&A

A domain-configurable document Q&A prototype: upload PDFs and ask questions in four specialised
prompt modes (enterprise SOP, legal, medical, study). FastAPI backend, React frontend, FAISS
vector search, Gemini embeddings and generation with a Groq fallback. It is an offline-validated
prototype, not a production-deployed service, and makes no claim of validated medical or legal accuracy.

> Resume-supporting work lives on branch `claude/youthful-gates-7szl31`
> (final checkpoint: see the latest commit on that branch). `main` holds the original version.

## Features (implemented)

- PDF upload → PyMuPDF text extraction → chunking → `gemini-embedding-001` → FAISS `IndexFlatL2`.
- Four domain system prompts; Gemini `gemini-2.0-flash` with fallback to Groq
  `llama-3.3-70b-versatile` when the Gemini call raises any exception (no timeout is set; an empty
  or poor answer does not trigger the fallback).
- Sources are the file names of all top-4 retrieved chunks, not citations verified against the
  answer; there is no abstention or grounding check for unsupported answers. CSV audit log per answered query.
- "Confidence" label (high/medium/low). **Caveat:** it is a chunk-length heuristic, not a
  relevance or calibration measure (known defect D-02).
- **Incremental indexing** — document fingerprints + content-addressed, namespaced embedding
  cache: only new or changed chunks are embedded; atomic, crash-detectable index publication.
- **Version-aware retrieval cache** — the FAISS index and chunk metadata are kept in memory as one
  validated generation and reloaded only when a new index is published.
- **Safe persistence** — chunk metadata is never loaded with an unrestricted unpickler; uploads are
  confined to the PDF directory and validated before they enter the corpus.
- **Optional structure-aware, token-aware chunking** (`DOCINTEL_CHUNKING=structure`): paragraph
  blocks packed within document sections up to ~192 tokens. Default remains 800-word windows.

## Architecture

```
React UI ──REST──> FastAPI  POST /upload ─> stage + validate PDF ─> build_index()
                            POST /chat   ─> retrieve() ─> llm.ask() ─> answer + sources + audit row
build_index(): fingerprint docs ─> reuse unchanged chunk texts ─> embed cache misses only
               ─> rebuild IndexFlatL2 ─> atomic publish (faiss.index, chunks.pkl, manifest.json)
retrieve():    validate cached generation (3 stat calls) ─> reload only if changed ─> top-4 L2 search
```

Key modules: `backend/ingest.py`, `backend/index_store.py`, `backend/embedding_cache.py`,
`backend/retriever.py`, `backend/structure_chunker.py`, `backend/llm.py`, `backend/main.py`.

## Measured results (offline; evidence committed in the repository)

| Result | What it is compared against | Evidence |
|---|---|---|
| Embedding requests for 100 sequential single-chunk uploads: **5,050 → 100 (−98.0 %)** | The **original full-corpus re-embedding** on every upload (not a competent baseline); offline fake embedder, exact call counts | `docs/phase1/bench/p1a_indexing.json` |
| Chunk-level reuse vs **document-level incremental indexing** (both skip unchanged and renamed files): structure-aware chunks need **87.2 % fewer** embeddings across 15 edit categories (62–97 % per category); default 800-word windows 31.9 % (0 % for beginning insertions); **0 % for unique documents** | Same chunker for both; 11 real QASPER papers; real local BGE compute 201 s → 22.5 s (structure-aware) | `evaluation/qasper/cache_benchmark.json` |
| Local warm-cache retrieval p50 **69.9 → 2.8 ms (25.1×)** at 20,000 × 768-d | **In-memory index reuse vs re-loading the index from disk on every query**, not a faster FAISS search; synthetic vectors (infrastructure, not retrieval quality) | `docs/phase1/bench/p1b_retrieval.json` |
| QASPER (38 papers, 123 questions): structure-aware vs fixed 256-token windows (overlap 32) at equal budget: evidence recall @1,024 context tokens **+0.077 [95 % CI 0.014, 0.144]**; top-4 context **−33 %** (995 → 664 tokens); MRR −0.013 [−0.073, 0.046] (not shown non-inferior) | Pre-registered, paper-clustered paired bootstrap; dataset paper text (not original PDFs); academic domain only | `evaluation/qasper/results_test.json` |

Experimental: all quality results use local `BAAI/bge-small-en-v1.5` (ONNX), not the application's
Gemini embeddings. Not measured: live Gemini embedding/generation inside the application, provider billing.

## Live answer-level pilot (QASPER, 40 paired questions)

Scope: **live Groq generation over local BGE retrieval on QASPER paper text** — not an evaluation of
the full Gemini–FAISS application (no separate Gemini credential was available; the app smoke test
was deferred). Generation `openai/gpt-oss-120b` (temperature 0, reasoning effort low, 1,024-token
completion budget), identical prompts for both configurations; judge `openai/gpt-oss-20b` via
RAGAS 0.4.4 Faithfulness. Evidence: `evaluation/qasper/live_report.json`, raw logs `live_*.jsonl`.

| | Fixed 256-token windows | Structure-aware 256 | Paired difference (95 % paper-clustered CI) |
|---|---|---|---|
| QASPER answer F1 (official scoring) | 0.441 | 0.366 | −0.074 [−0.174, +0.028] → **inconclusive** (frozen rule) |
| Provider prompt tokens / question | 1,107.6 | 768.0 | −340 [−390, −293] |
| RAGAS Faithfulness (matched scorable pairs, n = 11) | — | — | +0.212 [0.000, +0.417] (not shown better) |
| Abstentions (“Unanswerable”) | 13 / 40 (2 / 2 gold correct) | 12 / 40 (1 / 2 correct) | over-abstention: only 2 gold-unanswerable |
| End-to-end p50 / p95 | 1.19 s / 48.5 s | 1.43 s / 47.6 s | p95 dominated by free-tier rate-limit pacing |

Responses: 80/80 answered, 0 failed, 0 truncated (no budget escalations). Faithfulness: 25 judged,
13 skipped as model abstentions (pre-declared rule), 0 failed. Provider usage: 130 attempted
requests, 0 HTTP errors; 79,793 generation + 71,717 judge tokens. LangSmith: judge-run traces were
accepted (27/27 upload batches HTTP 202); **all generation-run traces were lost** because the
harness's upload-acknowledgement hook raised on compressed bodies (fixed afterwards); the
tracing-only key cannot read traces back. Answer F1 is not a faithfulness or hallucination measure.

## Application performance (offline; simulated provider)

| Change | Result | Evidence |
|---|---|---|
| Opt-in embedding batching (`DOCINTEL_EMBED_BATCH_SIZE`, 1–100; default 1 = original single-item path) | 49 QASPER papers reconstructed as PDFs (246 chunks): **246 → 8 calls to the mocked embedding provider at batch 32** (31 at 8, 16 at 16). These are simulator calls, not measured live HTTP requests; same 246 items, so no token/billing change; index bit-identical and top-4 identical to batch 1. Indexing wall time 32.0 s → 3.2 s under an **assumed** latency model (120 ms/request + 4 ms/item). Gemini batch-size limits and live compatibility are **unverified** | `docs/phase4/batch_bench.json` (field `http_requests` = simulator calls) |
| Endpoints run blocking work in the thread pool, bounded (`DOCINTEL_MAX_CONCURRENT_CHAT`=8, `DOCINTEL_MAX_CONCURRENT_UPLOADS`=2) | Simulated blocking provider (30 ms embed + 300 ms LLM), 40 requests per level: throughput 3.0 → 14.9 / 23.8 / 23.9 req/s at concurrency 5 / 10 / 20 (saturates at the configured bound of 8); event-loop lag max ~13.3 s → ≤ 2.8 ms; provider concurrency ≤ 8; 0 failures; audit rows intact | `docs/phase4/load_test.json` |
| Cache reconstruction | Index rebuilt from cached + new vectors vs full re-encode (real BGE, 5 edit types, 2 chunkers): **410/410 identical top-4 rankings** (10 configurations × 41 queries), max distance difference 0.0 | `evaluation/qasper/cache_topk_equivalence.json` |

Concurrency limits are **per process**: with N server workers up to N × `DOCINTEL_MAX_CONCURRENT_CHAT`
chats may call providers at once; index builds are serialised host-wide by the build lock, and an
upload's file replacement and rebuild happen under one lock hold. PyMuPDF (not thread-safe) runs
only on one dedicated worker thread, so ingestion is serialised while chats stay concurrent. Shared
SDK clients rely on their httpx connection pools being thread-safe (not independently verified).
Simulated-provider results are not live production latency. In the in-process load test, baseline
per-request client latency under-reports queueing (blocked client coroutines cannot start timers);
throughput and event-loop lag are the valid comparisons. Live-pilot follow-up (offline, post-hoc):
on the 25 questions both configurations answered, the F1 difference is −0.001 [−0.079, 0.097];
the −0.074 pilot gap comes from 5 questions with different abstention behaviour
(`evaluation/qasper/pilot_analysis.json`).

## Retrieval ablation (QASPER, local BGE harness — not application validation)

Structure-aware T=256 chunks, top-4, 1,024-token budget, pool 20; selection frozen on the 41 dev
questions (`evaluation/qasper/ablation_criteria.json`). Test results are confirmatory only (this
test set's retrieval results were examined in earlier phases).

| Config | Dev R@1024 tok | Dev MRR@10 | Test R@1024 tok | Test MRR@10 | Test R@4 / P@4 | Test p50 latency |
|---|---|---|---|---|---|---|
| A dense | 0.687 | 0.656 | 0.657 | 0.453 | 0.608 / 0.197 | 11 ms |
| B dense + BM25 (RRF) — eliminated on dev | 0.705 | 0.502 | 0.735 | 0.518 | 0.640 / 0.215 | 11 ms |
| C = B + cross-encoder rerank | 0.816 | 0.610 | 0.764 | 0.625 | 0.723 / 0.244 | 470 ms |
| **Selected: dense + cross-encoder rerank** | 0.816 | 0.610 | **0.772** | **0.624** | 0.714 / 0.240 | 447 ms |

Selected vs A on test (paired, paper-clustered 95 % CI): evidence recall @1,024 tokens
+0.114 [0.025, 0.208]; MRR@10 +0.172 [0.104, 0.244]; ~+435 ms CPU per query. BM25/RRF was
eliminated by the frozen dev rule although it helped on test. Available in the app as
`DOCINTEL_RERANKER=minilm` (default off; optional deps `onnxruntime`, `tokenizers`,
`huggingface_hub`; pool `DOCINTEL_RERANK_POOL`, default 20). Answer-level effect not yet measured.

## Evaluated and rejected (kept for transparency)

- **Hybrid BM25 + dense (RRF)** — best on dev, worse on held-out (MRR@10 0.792 vs dense 0.865);
  not deployed. `evaluation/results/RESULTS.md`
- **Cross-encoder reranking** — no quality gain on dev, ~16–20× latency; not selected.
- **Relevance-margin context pruning** — −51.5 % context tokens on an unseen set, but evidence
  retention 0.944 missed the pre-registered 0.95; not deployed. `evaluation/results/context_unseen.json`

## Reproduce

```bash
pip install -r requirements-test.txt        # backend runtime + test deps (unpinned)
python -m pytest                             # offline: no credentials, network blocked
scripts/run_frontend_tests.sh                # Vitest; does not modify frontend manifests
python scripts/mutation_check.py --json /tmp/mutation.json
python scripts/phase1_bench_indexing.py --reps 3 --out /tmp/p1a.json
python scripts/phase1_bench_retrieval.py --large --out /tmp/p1b.json
# real-model evaluations (need onnxruntime, tokenizers, huggingface_hub and Hugging Face access):
python evaluation/run_retrieval_eval.py --split dev
python evaluation/chunking/run_chunking_eval.py --split dev && python evaluation/chunking/run_chunking_eval.py --split test
# QASPER external benchmark (frozen selection in evaluation/qasper/frozen.json; needs pyarrow):
python evaluation/qasper/run_qasper_eval.py --split test
PYTHONPATH=. python evaluation/qasper/run_cache_benchmark.py        # B vs C caching by edit category
python evaluation/qasper/live_eval.py --dry-run                      # live pilot harness, no network
```

## Model configuration

The original generation defaults (`gemini-2.0-flash`, Groq `llama-3.3-70b-versatile`) have been
discontinued upstream and are kept only as code defaults for compatibility. Set replacements:

```bash
export DOCINTEL_GEMINI_MODEL=<current Gemini generation model>   # primary
export DOCINTEL_GROQ_MODEL=openai/gpt-oss-120b                    # fallback (check your Groq account)
```

The fallback to Groq runs when the Gemini call raises; the `model` response field reports the ID used.

## Run the app

```bash
printf 'GEMINI_API_KEY=...\nGROQ_API_KEY=...\n' > .env
PYTHONPATH=. uvicorn backend.main:app --reload     # http://127.0.0.1:8000
cd frontend && npm install && npm run dev          # http://localhost:5173
```

API: `GET /health`; `POST /upload` (multipart `file`); `POST /chat` `{"question", "mode"}` →
`{"answer", "model", "confidence", "sources"}`.

## Limitations

- Benchmarks are offline and single-machine. Retrieval quality is measured on public QASPER
  papers (38 test papers, 123 annotated questions; academic domain only, dataset text rather than
  original PDFs) plus small synthetic sets used for earlier experiments. Quality results use a
  local BGE model, not the application's Gemini embeddings.
- No authentication, upload size limit or multi-user isolation; locking is single-host (POSIX `flock`).
- Open defects are tracked in `docs/KNOWN_DEFECTS.md` (e.g. confidence heuristic, error masking in `/chat`).
- No pinned production `requirements.txt` yet.

## Author

**Shashwat Sharma** — [GitHub](https://github.com/shashwatsharma028-sys/docIntel-ai) ·
[LinkedIn](https://www.linkedin.com/in/shashwat-sharma-138b21213)

## License

MIT
