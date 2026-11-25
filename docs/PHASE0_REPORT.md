# Phase 0 Report — Baseline Preservation and Regression Testing

- Baseline checkpoint: `main` = `3e61052e89714a5af91ae686a82369c86d461b8e` (unchanged, not pushed to).
- Working branch: `claude/youthful-gates-7szl31`.
- Production code changed: **none** (`backend/`, `frontend/src/`, `frontend/package*.json` untouched).
- Live API calls made: **none**. Optimisation work started: **none** (awaiting gate G0).

## 1. Test results (exact)

| Suite | Command | Passed | Failed | Skipped | Expected-fail |
|---|---|---|---|---|---|
| Backend (pytest) | `python -m pytest` | 108 | 0 | 0 | 10 (strict xfail) |
| Frontend (Vitest + Testing Library, jsdom) | `scripts/run_frontend_tests.sh` | 11 | 0 | 0 | 2 (`it.fails`) |
| **Total** | | **119** | **0** | **0** | **12** |

All 12 expected failures were re-run as normal tests and confirmed to fail on their documented
assertion (not on setup errors). Strict mode means fixing a defect makes the suite fail until the
test is deliberately converted (verified with mutant M12).

### Coverage map

| Area | Tests |
|---|---|
| PDF text extraction | `test_ingest.py::TestParsePdf` (single/multi-page order, blank, malformed, missing) |
| Chunking & overlap | `TestChunkText` (window positions 0/700/1400, 100-word overlap, full word coverage, 50-char filter) |
| Embeddings (injected mock) | `TestEmbedding` — real `get_embedding` runs; only the SDK call is faked |
| FAISS creation & persistence | `TestBuildIndex` (type/dim/ntotal, row i = embedding of chunk i, pickle contents, rebuild, non-PDF ignored) |
| Retrieval ranking & source mapping | `test_retriever.py::TestRankingAndMapping` (hand-built geometry with independently derived order, exact-match mapping, intra-document chunk selection) |
| Golden retrieval | 8 queries × 4 synthetic domain documents, expectations defined in `tests/fixtures/golden/expected_retrieval.json` before running |
| Upload / Query API | `test_api.py` (contracts, validation 422s, overwrite, accumulation, CORS, health, route inventory) |
| Four domain modes | prompt identity, citation instruction, guardrails, exact Gemini & Groq payloads per mode, end-to-end via API, unknown/default mode |
| LLM fallback | Gemini OK (Groq not called), Gemini down → Groq, both down, via API |
| Confidence | contract (domain, empty → low) + `characterization` thresholds + D-02 xfail |
| Source citations | `build_prompt` exact format/order; API sources unique incl. multi-chunk document |
| Audit logging | row schema, ISO timestamp, truncation, CSV quoting round-trip, append, fallback logged, no row on total failure |
| Empty index / malformed docs / errors | empty-index response, missing chunks file, blank/corrupt PDFs, 422s |
| Frontend workflows | initial render, chat request/render, confidence colours, sources toggle & plural, error message, loading/double-send, Enter/Shift+Enter, mode switch clears chat, upload multipart/list/success/failure, PDF-only dropzone |

Offline guarantees: dummy keys forced, SDK classes patched before import, strict client mocks fail
on any unconfigured call, outbound sockets blocked, each test in its own temp working directory.

## 2. Coverage

| Scope | Lines | Branches |
|---|---|---|
| Backend (`pytest --cov=backend --cov-branch`) | 144/145 statements (99 %) — only `ingest.py:54` (`__main__` guard) missed | 15/16 |
| Frontend (`vitest --coverage`, v8, `src/**`) | 98.46 % lines, 95.83 % statements | 92.98 % |

Frontend misses: `main.jsx` (bootstrap), `Sidebar.jsx` drag-active styling and the status-timeout
callback. Coverage measures execution, not assertion strength — see mutation results.

## 3. Mutation testing (targeted)

`python scripts/mutation_check.py` injects one defect at a time into a throwaway copy of the repo.
**Result: 32/32 mutants detected** (26 backend, 6 frontend); full list in `mutation_results.json`.

- Mutants cover: overlap removal, window step off-by-one, short-chunk filter, chunk truncation,
  first-page-only extraction, wrong FAISS metric, swapped source/text (ingest and retriever),
  wrong embedding model, default `top_k`, reversed ranking, mode ignored by Gemini, Groq role,
  fallback label/target, audit truncation, audit removed, citations stripped, confidence boundary,
  empty-evidence confidence, source de-duplication, upload skipping indexing, API default mode,
  CORS widening, empty-index response; frontend: mode dropped from request, history kept on mode
  switch, confidence colour, pluralisation, Shift+Enter, uploaded file not listed.
- M12 deliberately **fixes** D-01; it is detected because the strict xfail turns into XPASS.
- First run found 1 survivor (M22, sources not de-duplicated): the golden corpus never retrieved two
  chunks from one file. A multi-chunk API test was **added** (no assertion was weakened); re-run 32/32.

## 4. Baseline performance (offline, `baseline_performance.json`)

Mocked embeddings/LLM; wall times exclude all network latency. Python 3.13, faiss-cpu 1.16, x86_64.

| Measurement | Result |
|---|---|
| `parse_pdf`, 20 pages / 8 000 words | 13.8 ms mean |
| `chunk_text`, 100 000 words (143 chunks) | 3.7 ms mean |
| Embedding calls, 10 sequential single-chunk uploads | per upload `1,2,…,10`; cumulative **55** (ideal 10) — O(n²), D-03 |
| `retrieve` d=768, n=1 000 / 5 000 chunks | 1.05 ms / 6.0 ms (disk reload 0.66 / 5.4 ms; FAISS search 0.18 / 0.98 ms) |
| `retrieve` d=3072, n=1 000 / 5 000 chunks | 3.3 ms / 51.4 ms (disk reload 2.2 / 44.5 ms; FAISS search 0.8 / 4.4 ms) |

In production the dominant costs are the per-chunk embedding API calls and LLM latency, which were
intentionally not measured (live API restriction). Re-run with `python scripts/baseline_perf.py --out <file>`
on the same machine to compare.

## 5. Known defects

16 defects (D-01…D-16) and 11 limitations recorded in `KNOWN_DEFECTS.md`; 11 have strict
expected-failure tests (12 test cases). High severity: D-01 FAISS `-1` ids, D-02 misleading
confidence, D-05 malformed upload poisons later uploads, D-08 path traversal.

## 6. Preserved functionality

All existing behaviour in `BASELINE.md` is preserved and pinned: three endpoints and their exact
request/response shapes, empty-index message, 4 domain prompts and fallback chain, prompt/citation
format, confidence values, audit-log format, chunking parameters, FAISS persistence layout, and all
frontend workflows.

## 7. Remaining limitations of this baseline

- Mocked tests verify wiring and contracts only; **no semantic retrieval-quality evaluation** with
  real Gemini embeddings exists (requires live calls, gate G5).
- No live smoke test of Gemini/Groq model availability (L-10).
- Frontend tests run in jsdom (no real browser, no CSS/layout); no end-to-end browser test against a
  running backend.
- Frontend test tooling is not declared in `package.json` (lockfile out of sync, D-12; manifest change
  needs gate G4). `scripts/run_frontend_tests.sh` installs it in a scratch directory instead; it needs
  npm registry access on first run.
- No production `requirements.txt` (D-11); `requirements-test.txt` is unpinned.
- No CI workflow added (not requested; would need approval).
- Performance figures are single-machine, offline, and exclude network latency.
- Concurrency (L-3) and pickle safety (L-4) are documented, not tested.

## 8. Blockers

- The optimisation specifications referred to as "previously supplied" are **not present** in this
  session; they must be re-supplied before Phase 1.
- Phase 1 must not start until this baseline passes independent review (gate G0).
