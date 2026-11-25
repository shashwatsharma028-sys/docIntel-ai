# DocIntel AI — Multi-Domain RAG Document Q&A

Upload PDFs and ask questions about them in four domain modes (enterprise SOP, legal, medical,
study). FastAPI backend, React frontend, FAISS vector search, Gemini embeddings and generation
with a Groq fallback.

> Resume-supporting work lives on branch `claude/youthful-gates-7szl31`
> (final checkpoint: see the latest commit on that branch). `main` holds the original version.

## Features (implemented)

- PDF upload → PyMuPDF text extraction → chunking → `gemini-embedding-001` → FAISS `IndexFlatL2`.
- Four domain system prompts; Gemini `gemini-2.0-flash` with automatic fallback to Groq
  `llama-3.3-70b-versatile` when the Gemini call raises.
- Document-level source list per answer; CSV audit log of every answered query.
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

| Result | Conditions | Evidence |
|---|---|---|
| Embedding requests for 100 sequential single-chunk uploads: **5,050 → 100 (−98.0 %)**; per-upload indexing p50 66.6 → 5.6 ms | Offline fake embedder (exact call counts, zero network latency); 3 alternating repetitions; baseline = original code loaded from git | `docs/phase1/bench/p1a_indexing.json` |
| Local warm-cache retrieval p50 **69.9 → 2.8 ms (25.1×)** at 20,000 × 768-d; 7.1–15.2× at 1k–5k chunks; identical results 50/50 queries | Precomputed query vectors, no network/LLM time; 4 alternating rounds | `docs/phase1/bench/p1b_retrieval.json` |
| Structure-aware chunking: **−57 % top-4 context tokens** (1,149.5 → 493.8) at equal evidence recall@4 (1.000) and MRR 0.875 → 0.896 vs dev-selected fixed token windows | Real PDF extraction, local `BAAI/bge-small-en-v1.5`, 16 fresh frozen questions with span-level evidence; acceptance rule pre-registered | `evaluation/chunking/results/test.json` |

Not measured: live Gemini/Groq latency, provider billing, answer quality.

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
```

## Run the app

```bash
printf 'GEMINI_API_KEY=...\nGROQ_API_KEY=...\n' > .env
PYTHONPATH=. uvicorn backend.main:app --reload     # http://127.0.0.1:8000
cd frontend && npm install && npm run dev          # http://localhost:5173
```

API: `GET /health`; `POST /upload` (multipart `file`); `POST /chat` `{"question", "mode"}` →
`{"answer", "model", "confidence", "sources"}`.

## Limitations

- Benchmarks are offline and single-machine; evaluation corpora are small and synthetic
  (9 documents; 16–24 questions per split). Quality results use a local model, not Gemini.
- No authentication, upload size limit or multi-user isolation; locking is single-host (POSIX `flock`).
- Open defects are tracked in `docs/KNOWN_DEFECTS.md` (e.g. confidence heuristic, error masking in `/chat`).
- No pinned production `requirements.txt` yet.

## Author

**Shashwat Sharma** — [GitHub](https://github.com/shashwatsharma028-sys/docIntel-ai) ·
[LinkedIn](https://www.linkedin.com/in/shashwat-sharma-138b21213)

## License

MIT
