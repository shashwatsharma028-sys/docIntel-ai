# CLAUDE.md — DocIntel AI Engineering Contract

This file is binding for any automated agent (Claude Code or otherwise) and for human
contributors working on this repository. When in doubt, stop and ask the repository owner.

## 1. Repository protection

| Rule | Detail |
|---|---|
| Protected baseline | `main` is frozen at the verified checkpoint `3e61052e89714a5af91ae686a82369c86d461b8e`. Do not push, merge, rebase, reset or force-push `main`. |
| Working branch | All work happens on the isolated branch `claude/youthful-gates-7szl31` (or another branch the owner names). |
| History | Never rewrite published history (no force-push, no interactive rebase, no amend of pushed commits). |
| Merges / PRs | Do not merge anything. Do not open a pull request unless the owner explicitly asks. |
| Commit separation | Baseline tests and documentation are committed separately from optimisation/behaviour changes. One logical change per commit. |

## 2. Behavioural preservation

- Never silently change existing functionality, public API contracts (`GET /health`,
  `POST /upload`, `POST /chat` request/response shapes — see `docs/BASELINE.md §3`),
  audit-log format, or frontend behaviour.
- Do not silently fix known defects. Every defect in `docs/KNOWN_DEFECTS.md` is fixed only in a
  dedicated, approved change that cites its ID (e.g. `D-01`) and flips the matching expected-failure
  test into a normal test in the same commit.
- Do not modify `frontend/package.json` or `frontend/package-lock.json` without approval
  (the lockfile is already out of sync — `D-12`).

## 3. Testing requirements

- Backend: `python -m pytest` (config in `pytest.ini`; deps in `requirements-test.txt`).
- Frontend: `scripts/run_frontend_tests.sh` (does not touch frontend manifests).
- Mutation check: `python scripts/mutation_check.py` — all mutants must be detected.
- Offline baseline performance: `python scripts/baseline_perf.py --out <file>`.
- **All default tests run offline** with no credentials. `tests/conftest.py` forces dummy API keys,
  replaces the Gemini/Groq SDK clients with strict mocks that fail on unconfigured use, and blocks
  outbound sockets. Never weaken these guards.
- After every optimisation: run the targeted tests for the touched module, then the full backend
  and frontend suites, then the mutation check. Report exact pass/fail/skip/xfail counts.
- **Never weaken, delete, skip or rewrite an existing assertion to make a change pass.** If an
  assertion must change, it is an API/behaviour change and needs owner approval first.
- Test markers:
  - `known_defect` + `xfail(strict=True)` — desired behaviour the baseline does not meet. A strict
    XPASS fails the suite, so fixing a defect is always a visible, deliberate act.
  - `characterization` — pins current behaviour (possibly imperfect) for change detection only. It is
    not a statement that the behaviour is desirable.
- Mocked tests prove wiring and contracts, **not** semantic retrieval quality. Do not present mocked
  retrieval results as quality evaluation. Real-embedding evaluation is a separate, opt-in activity
  that requires owner approval (it calls live APIs).
- Test fixtures live under `tests/` and `frontend/tests/` only — never in production code paths.

## 4. Security rules

- Never commit secrets. `.env` is gitignored; do not print, log or echo API keys.
- Do not make live Gemini/Groq (or any external) API calls from tests, scripts or CI without explicit
  owner approval for that specific run. Default to mocks.
- Do not send repository content, user data or credentials to third-party services.
- Treat uploaded documents and the audit log (`data/audit_log.csv`, may contain medical/legal text)
  as sensitive; never commit anything under `data/`.
- Known security defects (e.g. `D-08` path traversal) are fixed only via an approved change.

## 5. Approval gates

| Gate | Requires owner approval before proceeding |
|---|---|
| G0 | Phase 0 baseline reviewed and accepted (independent review). **No optimisation work before G0.** |
| G1 | Each optimisation specification, before implementation. |
| G2 | Any change to a public API contract, audit-log format or visible frontend behaviour. |
| G3 | Fixing any known defect (cite the `D-xx` ID). |
| G4 | Dependency additions/upgrades, manifest or lockfile changes. |
| G5 | Any live external API call (evaluation, smoke test). |
| G6 | Merging, PR creation, or any action on `main`. |

## 6. Project map

- `backend/ingest.py` — PDF parse (PyMuPDF), word chunking (800/100), Gemini embeddings, FAISS build.
- `backend/retriever.py` — loads FAISS index + pickled chunks, top-k L2 search.
- `backend/llm.py` — 4 domain prompts, Gemini → Groq fallback, confidence heuristic, CSV audit log.
- `backend/main.py` — FastAPI app and CORS.
- `frontend/src` — React 19 + Vite 8 + Tailwind 3 UI.
- `docs/BASELINE.md`, `docs/KNOWN_DEFECTS.md`, `docs/PHASE0_REPORT.md` — Phase 0 baseline record.
