# Phase 1 — Test Status Change Log

Every change in test status, with the exact reason. Original assertions of converted
tests are retained verbatim; only the `known_defect` / `xfail(strict=True)` markers are removed.

| Commit scope | Test | Before | After | Reason |
|---|---|---|---|---|
| fix D-01 | `tests/test_retriever.py::TestFixedDefects::test_fewer_chunks_than_top_k_returns_only_real_chunks` | strict xfail (`TestKnownDefects`) | pass | `retrieve` now discards FAISS `-1` ids (`0 <= i < len(chunks)`). Class renamed `TestKnownDefects` → `TestFixedDefects`; assertions unchanged. Mutant M12 changed from "apply the fix" to "reintroduce `i < len(chunks)`" (same detection intent). |
