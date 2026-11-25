# Known Defects and Limitations (baseline `3e61052`)

None of these were fixed in Phase 0. Each fix needs approval (CLAUDE.md gate G3) and must flip the
listed expected-failure test to a normal passing test in the same commit. "Test" = the strict
expected-failure test that currently documents the desired behaviour.

Severity: **H** high (data integrity / security / misleading users) · **M** medium · **L** low.

## Functional and security defects

| ID | Sev | Location | Defect | Test |
|---|---|---|---|---|
| D-01 | H | `retriever.py:23-24` | When the index holds fewer vectors than `top_k`, FAISS returns `-1` ids. The guard `i < len(chunks)` admits `-1`, so `chunks[-1]` (the last chunk) is returned as fake extra hits, duplicating context sent to the LLM. Any corpus with < 4 chunks is affected. | `test_retriever.py::TestKnownDefects::test_fewer_chunks_than_top_k_returns_only_real_chunks` |
| D-02 | H | `llm.py:21-29` | "Confidence" is the mean character length of retrieved chunks. It ignores retrieval distance and relevance, so irrelevant long chunks score `high`. Misleading for legal/medical modes. | `test_llm.py::TestConfidenceKnownDefects::test_irrelevant_long_chunks_are_not_high_confidence` |
| D-03 | M | `main.py:29`, `ingest.py:33-50` | Every upload re-parses and re-embeds **all** PDFs. Embedding calls for n sequential uploads grow O(n²) (measured: 55 calls for 10 single-chunk uploads vs 10 needed). Also the roadmap item in README. | `test_api.py::TestKnownDefects::test_upload_embeds_only_new_document` |
| D-04 | M | `main.py:43-44` | `/chat` catches every exception and answers "No documents indexed yet. Please upload a PDF first." with HTTP 200 — including LLM outages and embedding failures. Errors are hidden (the exception is not logged). | `test_api.py::TestKnownDefects::test_llm_outage_is_not_reported_as_missing_documents` |
| D-05 | H | `main.py:25-29` | A malformed PDF is written to `data/pdfs` before indexing; indexing then raises (HTTP 500) and the bad file stays, so **every later upload also fails** until it is removed manually. | `test_api.py::TestKnownDefects::test_malformed_upload_does_not_break_later_uploads` |
| D-06 | L | `main.py:23-30` | No server-side file-type validation. Non-PDF files are saved and reported "uploaded and indexed successfully" although the indexer ignores them. | `test_api.py::TestKnownDefects::test_non_pdf_upload_is_not_reported_as_indexed` |
| D-07 | L | `ingest.py:25-31` | Chunk windows step by 700; when the text length is 701–800 words past a window start, a final window wholly inside the previous chunk's overlap is emitted as a redundant chunk (e.g. 750 or 800 words → 2 chunks, second a subset of the first). | `test_ingest.py::TestChunkingKnownDefects::test_no_chunk_is_wholly_contained_in_previous[750,800]` |
| D-08 | H | `main.py:26` | **Path traversal**: `file.filename` is used unsanitised, so `../x.pdf` writes outside `data/pdfs` (verified: writes `data/x.pdf`). | `test_api.py::TestKnownDefects::test_upload_cannot_write_outside_pdf_folder` |
| D-09 | M | `ingest.py:40-42`, `main.py:30` | A blank or image-only (scanned) PDF yields no text; `build_index` prints "No PDFs found." and returns without writing, yet the API reports "indexed successfully". No OCR. If an indexed PDF is overwritten by a blank one of the same name and no other PDFs exist, the stale index is not replaced. | `test_api.py::TestKnownDefects::test_blank_pdf_upload_is_not_reported_as_indexed` |
| D-13 | M | `App.jsx:14-21` | Frontend upload ignores `res.ok`: an HTTP error with a JSON body is shown as "Indexed successfully!" and the file is listed. | `frontend/tests/app.test.jsx` › known defects › D-13 (`it.fails`) |
| D-14 | L | `Sidebar.jsx:76-80` | Upload failure text ("Upload failed") is rendered with the green success icon/colour. | `frontend/tests/app.test.jsx` › known defects › D-14 (`it.fails`) |

## Build, packaging and documentation defects (documented, no runtime test)

| ID | Sev | Defect |
|---|---|---|
| D-10 | M | Backend modules read API keys at import (`os.environ[...]` → `KeyError` without keys) and create three separate Gemini clients. Importing any module requires credentials; tests work around this with dummy keys + patched SDK classes. |
| D-11 | M | README install step uses `requirements.txt`, which does not exist; no pinned backend dependency set. README also states "800 tokens" (chunking is by whitespace words), a 768-d embedding (not set in code; the actual dimension is whatever the API returns — unverified offline), and React 18 (package.json has React 19). Root `package.json` declares `tailwindcss ^4` while `frontend` uses `^3.4`. |
| D-12 | M | `frontend/package-lock.json` is out of sync with `frontend/package.json` (`npm ci` fails with EUSAGE; plain `npm install` rewrites the lock). |
| D-15 | L | Stray empty file `frontend/src/frontend/src/components/frontend/src/components/Sidebar.jsx`. |
| D-16 | L | Frontend `eslint .` reports 2 pre-existing errors (unused `e` in `App.jsx:22`, `App.jsx:47`). |

## Limitations / observations (not regarded as defects; optimisation or design candidates)

- **L-1 Per-query disk reload**: `retrieve` re-reads the FAISS index and unpickles all chunks on every
  query (≈85–90 % of offline retrieval time at 1k–5k chunks; see `baseline_performance.json`).
- **L-2 Unbatched embeddings**: one embedding API call per chunk, sequential.
- **L-3 Blocking work in `async def`**: `build_index`, embedding and LLM calls are synchronous inside
  async endpoints and block the event loop; concurrent uploads race on the same index files.
- **L-4 Pickle persistence**: `chunks.pkl` is loaded with `pickle`; a tampered file executes code.
- **L-5 Audit log privacy/format**: stores raw questions (possibly medical/legal data) in plaintext,
  no header, no rotation; always appends `"..."`; source order is non-deterministic (`set`).
- **L-6 Sources are document-level only** (no page/chunk citation); order is non-deterministic.
- **L-7 CORS** allows only `http://localhost:5173`; opening the UI at `http://127.0.0.1:5173` is
  blocked. API URL `http://127.0.0.1:8000` is hard-coded in the frontend.
- **L-8 UI document list is client-side** and resets on reload while the server index persists.
- **L-9 No authentication, upload size limit, or per-user isolation.**
- **L-10 Model availability**: `gemini-2.0-flash` / `llama-3.3-70b-versatile` availability was not
  verified (live calls are out of scope).
- **L-11 Unknown modes** silently use the enterprise prompt; empty questions are accepted.
