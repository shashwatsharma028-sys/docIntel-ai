# DocIntel AI — Baseline Behaviour (Phase 0)

Recorded at checkpoint `3e61052e89714a5af91ae686a82369c86d461b8e` (`main`). This document
describes what the code **does**, not what the README claims. Divergences are listed in
`KNOWN_DEFECTS.md`. Nothing in the production code was changed in Phase 0.

## 1. Components

| File | Responsibility |
|---|---|
| `backend/ingest.py` | `parse_pdf` (PyMuPDF, pages joined with `\n`), `chunk_text` (whitespace words, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, step 700, drops chunks ≤ 50 characters), `get_embedding` (`gemini-embedding-001`, one call per chunk), `build_index` (re-reads **every** `*.pdf` in `data/pdfs`, builds `faiss.IndexFlatL2`, writes `data/index/faiss.index` + `data/index/chunks.pkl` = list of `(filename, chunk_text)`). |
| `backend/retriever.py` | `retrieve(query, top_k=4)`: reads index and pickle from disk on every call, embeds query, returns `[{"source", "text"}]` ordered by ascending L2 distance. |
| `backend/llm.py` | `SYSTEM_PROMPTS` for `enterprise`, `legal`, `medical`, `study` (unknown mode → enterprise). `ask()` tries Gemini `gemini-2.0-flash`, on any exception falls back to Groq `llama-3.3-70b-versatile` (`max_tokens=1000`). Returns `{answer, model, confidence}` and appends an audit row. |
| `backend/main.py` | FastAPI app, CORS for `http://localhost:5173` only. |
| `frontend/src` | React UI: Sidebar (mode selector, PDF dropzone, indexed-files list), Chat (input, Enter to send, Shift+Enter newline), Message (answer, confidence colour, collapsible sources, model). |

All paths are relative to the **process working directory** (`data/...`). Each backend module
creates its SDK client at import time and requires `GEMINI_API_KEY` (and `llm.py` also
`GROQ_API_KEY`) in the environment or `.env`.

## 2. Dependencies

- Backend (no requirements file in repo — see D-11): `pymupdf`, `faiss-cpu`, `numpy`,
  `google-genai`, `groq`, `fastapi`, `python-multipart`, `python-dotenv`, `uvicorn`.
  Phase 0 verified against pymupdf 2.1, faiss-cpu 1.16.0, google-genai 3.6.1, groq 1.7.0,
  fastapi 0.165.0, numpy 2.5.5 on Python 3.13.
- Frontend (`frontend/package.json`): react/react-dom 19.2, react-dropzone 15, lucide-react 1.16,
  vite 8, tailwindcss 3.4. `axios` and `react-markdown` are declared but unused.

## 3. API contracts (preserve exactly)

| Endpoint | Request | Success response |
|---|---|---|
| `GET /health` | — | `200 {"status": "ok"}` |
| `POST /upload` | multipart, field `file` (required, else 422) | `200 {"message": "<filename> uploaded and indexed successfully"}`; file saved to `data/pdfs/<filename>` (same name overwrites); full re-index. Malformed PDF → unhandled exception → `500`. |
| `POST /chat` | JSON `{"question": str (required), "mode": str = "enterprise"}`; missing question/invalid JSON → 422 | `200 {"answer": str, "model": "gemini-2.0-flash" \| "groq-llama-3.3", "confidence": "high" \| "medium" \| "low", "sources": [unique filenames of the retrieved chunks]}` |
| `POST /chat` (any exception: no index, embedding failure, both LLMs down) | | `200 {"answer": "No documents indexed yet. Please upload a PDF first.", "model": "", "confidence": "low", "sources": []}` |

LLM input: Gemini receives `"<system prompt>\n\n" + build_prompt(...)`; Groq receives
`[{"role":"system", prompt}, {"role":"user", build_prompt(...)}]`, where
`build_prompt = "Context:\n" + "\n\n".join("[Source: <file>]\n<text>") + "\n\nQuestion: <q>"`.

Confidence (current heuristic, see D-02): mean chunk length in characters > 300 → high,
> 150 → medium, else low; no chunks → low.

Audit log `data/audit_log.csv` (append, no header): `[ISO timestamp, mode, question,
answer[:100] + "...", ", ".join(unique sources)]`. Written only when an answer is produced.

## 4. Frontend workflows

- Initial: enterprise mode selected; empty-chat prompt; no indexed list.
- Chat: `POST http://127.0.0.1:8000/chat` with `{question, mode}`; user bubble, "Thinking..." while
  pending, double-send blocked, input cleared; answer bubble with `● <level> confidence`
  (green/yellow/red), `N source(s)` toggle and `· <model>`. Network error → "Something went wrong. Is
  the backend running?".
- Mode switch: selects the mode and clears the conversation.
- Upload: dropzone accepts PDFs only, single file; `POST /upload` multipart `file`; on resolved fetch
  the filename is appended to "Indexed Documents" and "Indexed successfully!" shows for 3 s; on fetch
  rejection "Upload failed". The list is client-side only and resets on reload.

## 5. Mocked vs real retrieval

The offline suite uses `tests/fakes.FakeEmbedder`, a hashed bag-of-words embedder. Golden retrieval
tests (`tests/fixtures/golden/expected_retrieval.json`) therefore prove that chunking, vector
storage, FAISS search, and id → chunk/source mapping are wired correctly. They say **nothing** about
the semantic quality of Gemini embeddings. A real-embedding evaluation requires live API calls and
is gated (CLAUDE.md G5).

## 6. Baseline quality state

- Frontend `eslint .`: 2 pre-existing errors (`App.jsx:22`, `App.jsx:47`, unused `e`). `vite build`
  succeeds.
- Test, coverage, mutation and performance results: see `PHASE0_REPORT.md`.
