# Phase 1 — Test Status Change Log

Every change in test status, with the exact reason. Original assertions of converted
tests are retained verbatim; only the `known_defect` / `xfail(strict=True)` markers are removed.

| Commit scope | Test | Before | After | Reason |
|---|---|---|---|---|
| fix D-01 | `tests/test_retriever.py::TestFixedDefects::test_fewer_chunks_than_top_k_returns_only_real_chunks` | strict xfail (`TestKnownDefects`) | pass | `retrieve` now discards FAISS `-1` ids (`0 <= i < len(chunks)`). Class renamed `TestKnownDefects` → `TestFixedDefects`; assertions unchanged. Mutant M12 changed from "apply the fix" to "reintroduce `i < len(chunks)`" (same detection intent). |
| fix D-08 | `tests/test_api.py::TestKnownDefects::test_upload_cannot_write_outside_pdf_folder` | strict xfail | pass | Upload stores only the final path component (`\` treated as a separator), confines the resolved path to `data/pdfs` (rejects symlink escape) and returns 400 for names reducing to empty/`.`/`..`/NUL. Assertions unchanged. |
| fix D-08 | `tests/test_phase1_upload_safety.py` (13 new) | — | pass | New targeted tests. Contract note: ordinary filenames are unchanged; for names with directory parts the success message now reports the stored basename. Names that previously raised 500 (`..`, `a/b.pdf` into a missing dir) now give 400 / store the basename. |
| fix D-05 | `tests/test_api.py::TestKnownDefects::test_malformed_upload_does_not_break_later_uploads` | strict xfail | pass | Uploads are staged as `.upload-<uuid>.partial` (ignored by the indexer); `.pdf` uploads are parsed before `os.replace` into the corpus; staged files are always removed. A malformed PDF never enters `data/pdfs` or replaces a good file. Error status (500) and non-PDF behaviour (D-06) unchanged. Assertions unchanged. |
| fix D-05 | `tests/test_phase1_upload_safety.py` (+4) | — | pass | Malformed upload leaves corpus/index untouched; malformed same-name replacement keeps original; a valid file whose indexing failed is re-indexed by the next upload; non-PDF behaviour pinned unchanged. |
