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
Gemini embeddings. Not measured: live Gemini/Groq latency, provider billing, answer quality.

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
