#!/usr/bin/env python3
"""Targeted mutation testing for the Phase 0 baseline suite.

Each mutant is a single, deliberate defect injected into a THROWAWAY COPY of the
repository (the working tree is never modified). The relevant suite is run
against the copy; a mutant is "killed" when the suite fails. Surviving mutants
show assertions that do not detect that defect.

Classification (Phase 1 hardening):
  killed    -- the suite ran and >=1 test FAILED (genuine assertion detection)
  survived  -- the suite passed with the mutant applied
  error     -- collection/import/setup/internal error, no test failure: NOT counted as a kill
  obsolete  -- the target text no longer occurs exactly once (reported, never dropped)
  not_run   -- runner unavailable (e.g. no frontend tooling); reported, never dropped
The denominator is always the full list of defined mutants.

Copies exclude .git, data/, virtualenvs, .env*, key/credential files, audit logs,
node_modules (symlinked read-only), caches and build output.

usage:  python scripts/mutation_check.py [--backend-only] [--json OUT]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (id, runner, file, old, new, description, expect)   expect: "killed"
MUTANTS = [
    ("M01", "py", "backend/ingest.py", "CHUNK_OVERLAP = 100", "CHUNK_OVERLAP = 0", "remove chunk overlap", "killed"),
    ("M02", "py", "backend/ingest.py", "i += CHUNK_SIZE - CHUNK_OVERLAP", "i += CHUNK_SIZE - CHUNK_OVERLAP + 1", "off-by-one window step", "killed"),
    ("M03", "py", "backend/ingest.py", "MIN_CHUNK_CHARS = 50", "MIN_CHUNK_CHARS = 0", "drop short-chunk filter", "killed"),
    ("M04", "py", "backend/ingest.py", "words[i:i + CHUNK_SIZE]", "words[i:i + CHUNK_SIZE - 1]", "chunk loses last word", "killed"),
    ("M05", "py", "backend/ingest.py", "page.get_text() for page in doc", "page.get_text() for page in list(doc)[:1]", "extract first page only", "killed"),
    ("M06", "py", "backend/index_store.py", "index = faiss.IndexFlatL2(", "index = faiss.IndexFlatIP(", "wrong FAISS metric/index type", "killed"),
    ("M07", "py", "backend/ingest.py", "all_chunks.extend((filename, t) for t", "all_chunks.extend((t, filename) for t", "swap source/text in persisted chunks", "killed"),
    ("M08", "py", "backend/ingest.py", 'EMBEDDING_MODEL = "gemini-embedding-001"', 'EMBEDDING_MODEL = "text-embedding-004"', "wrong embedding model (ingest)", "killed"),
    ("M09", "py", "backend/retriever.py", 'chunks[i][0], "text": chunks[i][1]', 'chunks[i][1], "text": chunks[i][0]', "source mapping swapped", "killed"),
    ("M10", "py", "backend/retriever.py", "top_k: int = 4", "top_k: int = 3", "default top_k changed", "killed"),
    ("M11", "py", "backend/retriever.py", "for i in indices[0]", "for i in indices[0][::-1]", "ranking order reversed", "killed"),
    ("M12", "py", "backend/retriever.py", "if 0 <= i < len(chunks)", "if i < len(chunks)", "reintroduce D-01 negative-index defect", "killed"),
    ("M13", "py", "backend/llm.py", 'system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["enterprise"])\n    response = gemini_client', 'system_prompt = SYSTEM_PROMPTS["enterprise"]\n    response = gemini_client', "Gemini ignores domain mode", "killed"),
    ("M14", "py", "backend/llm.py", '{"role": "system", "content": system_prompt}', '{"role": "user", "content": system_prompt}', "Groq system prompt sent as user", "killed"),
    ("M15", "py", "backend/llm.py", 'model_used = "groq-llama-3.3"', 'model_used = "gemini-2.0-flash"', "fallback mislabels model", "killed"),
    ("M16", "py", "backend/llm.py", "answer = ask_groq(chunks, question, mode)", "answer = ask_gemini(chunks, question, mode)", "fallback retries Gemini instead of Groq", "killed"),
    ("M17", "py", "backend/llm.py", "answer[:100]", "answer[:99]", "audit truncation length", "killed"),
    ("M18", "py", "backend/llm.py", "if avg_len > 300", "if avg_len >= 300", "confidence threshold boundary", "killed"),
    ("M19", "py", "backend/llm.py", '    if not chunks:\n        return "low"', '    if not chunks:\n        return "medium"', "empty evidence not low", "killed"),
    ("M20", "py", "backend/llm.py", "    log_query(question, answer, mode, sources)\n", "", "audit logging removed", "killed"),
    ("M21", "py", "backend/llm.py", "[Source: {c['source']}]", "[Source]", "citations stripped from prompt", "killed"),
    ("M22", "py", "backend/main.py", '"sources": list(set(c["source"] for c in chunks))', '"sources": [c["source"] for c in chunks]', "API sources not de-duplicated", "killed"),
    ("M23", "py", "backend/main.py", "    build_index()\n", "", "upload skips indexing", "killed"),
    ("M24", "py", "backend/main.py", 'mode: str = "enterprise"', 'mode: str = "legal"', "API default mode changed", "killed"),
    ("M25", "py", "backend/main.py", 'allow_origins=["http://localhost:5173"]', 'allow_origins=["*"]', "CORS widened", "killed"),
    ("M26", "py", "backend/main.py", '"model": "", "confidence": "low"', '"model": "", "confidence": "high"', "empty-index response contract", "killed"),
    # ---- Phase 1 / P1A: incremental indexing, cache validity, atomic publication, safe metadata
    ("P01", "py", "backend/ingest.py", '"model": EMBEDDING_MODEL,', '"model": "fixed",', "cache namespace ignores model identity", "killed"),
    ("P02", "py", "backend/ingest.py", 'f"{file_sha}|{chunker_signature()}"', 'f"{filename}|{chunker_signature()}"', "document reuse keyed by name, not content", "killed"),
    ("P03", "py", "backend/ingest.py", 'previous[0].get("corpus_fingerprint") == fingerprint', 'previous is not None', "unchanged-corpus check ignores content", "killed"),
    ("P04", "py", "backend/embedding_cache.py", "(k for k in self._vec if k not in live_keys)", "(k for k in self._vec)", "eviction may drop live entries", "killed"),
    ("P05", "py", "backend/ingest.py", 'stats["cache"] = cache.save({text_key(t) for _, t in all_chunks})', 'stats["cache"] = cache.stats()', "embedding cache never persisted", "killed"),
    ("P06", "py", "backend/index_store.py", "os.replace(backups.pop(final), final)", "backups.pop(final)", "publish rollback disabled", "killed"),
    ("P07", "py", "backend/index_store.py", 'raise pickle.UnpicklingError(f"refusing to load global {module}.{name} from chunk metadata")', "return super().find_class(module, name)", "unrestricted unpickling of metadata", "killed"),
    ("P08", "py", "backend/index_store.py", "        if manifest is not None:\n            _check_artifact(INDEX_PATH", "        if False:\n            _check_artifact(INDEX_PATH", "generation identity check skipped on load", "killed"),
    ("P09", "py", "backend/ingest.py", "            stats[\"embeddings_computed\"] += 1  # count every", "            pass  # count every", "embedding invocations not counted", "killed"),
    ("P10", "py", "backend/ingest.py", "            stats[\"published\"] = index_store.publish_empty()", "            stats[\"published\"] = False", "empty corpus leaves stale generation (D-09a)", "killed"),
    ("P11", "py", "backend/main.py", "        if name.endswith(\".pdf\"):\n            parse_pdf(staged)", "        if False:\n            parse_pdf(staged)", "upload validation skipped (D-05)", "killed"),
    ("P12", "py", "backend/main.py", 'name = os.path.basename((filename or "").replace("\\\\", "/"))', 'name = (filename or "")', "upload path not confined (D-08)", "killed"),
    ("F01", "js", "frontend/src/App.jsx", "JSON.stringify({ question, mode })", "JSON.stringify({ question })", "chat request drops mode", "killed"),
    ("F02", "js", "frontend/src/App.jsx", "    setMode(newMode)\n    setMessages([])", "    setMode(newMode)", "mode switch keeps history", "killed"),
    ("F03", "js", "frontend/src/components/Message.jsx", 'high: "text-green-400"', 'high: "text-red-400"', "confidence colour mapping", "killed"),
    ("F04", "js", "frontend/src/components/Message.jsx", 'message.sources.length > 1 ? "s" : ""', '"s"', "source pluralisation", "killed"),
    ("F05", "js", "frontend/src/components/Chat.jsx", 'e.key === "Enter" && !e.shiftKey', 'e.key === "Enter"', "Shift+Enter sends", "killed"),
    ("F06", "js", "frontend/src/App.jsx", "setUploadedFiles((prev) => [...prev, file.name])", "setUploadedFiles((prev) => prev)", "uploaded file not listed", "killed"),
]


# Never copy secrets, private data, audit logs, environments or build outputs into mutant copies.
_EXCLUDE_TOP = {".git", "data", "venv", ".venv", "env", "htmlcov", ".pytest_cache", "coverage"}
_EXCLUDE_ANY = {"node_modules", "__pycache__", ".pytest_cache", "coverage", "dist", ".coverage",
                "audit_log.csv", ".DS_Store"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _ignore(directory, names):
    rel = Path(directory).resolve().relative_to(ROOT)
    out = set()
    for n in names:
        if (rel == Path(".") and n in _EXCLUDE_TOP) or n in _EXCLUDE_ANY:
            out.add(n)
        elif n == ".env" or n.startswith(".env.") or n.endswith(_SECRET_SUFFIXES) or n.startswith("credentials"):
            out.add(n)
    return out


def copy_repo(dst: Path):
    shutil.copytree(ROOT, dst, ignore=_ignore, symlinks=True)
    nm = ROOT / "frontend" / "node_modules"
    if nm.exists():
        os.symlink(nm.resolve(), dst / "frontend" / "node_modules")
    leaked = [p for p in dst.rglob("*") if p.name == ".env" or p.name == "audit_log.csv"
              or (p.parent == dst and p.name in _EXCLUDE_TOP)]
    if leaked:
        raise SystemExit(f"refusing to run: excluded paths copied: {leaked}")


def _classify_pytest(rc: int, junit: Path) -> tuple[str, dict]:
    """killed = >=1 genuine test failure. Collection/setup/internal errors are NOT kills."""
    info = {"exit_code": rc, "failures": 0, "errors": 0, "tests": 0, "failed_tests": []}
    if junit.exists():
        root = ET.parse(junit).getroot()
        for suite in root.iter("testsuite"):
            info["failures"] += int(suite.get("failures", 0))
            info["errors"] += int(suite.get("errors", 0))
            info["tests"] += int(suite.get("tests", 0))
        for case in root.iter("testcase"):
            if case.find("failure") is not None:
                info["failed_tests"].append(f"{case.get('classname')}::{case.get('name')}")
    if rc == 0:
        return "survived", info
    if rc == 1 and info["failures"] > 0:
        return "killed", info
    return "error", info  # rc 2 collection/interrupt, 3 internal, 4 usage, 5 no tests, or setup errors only


def _classify_vitest(rc: int, report: Path) -> tuple[str, dict]:
    info = {"exit_code": rc, "failures": 0, "errors": 0, "tests": 0, "failed_tests": []}
    if report.exists():
        data = json.loads(report.read_text())
        info["failures"] = data.get("numFailedTests", 0)
        info["tests"] = data.get("numTotalTests", 0)
        for suite in data.get("testResults", []):
            failed = [a for a in suite.get("assertionResults", []) if a.get("status") == "failed"]
            info["failed_tests"] += [a.get("fullName") for a in failed]
            if suite.get("status") == "failed" and not failed:
                info["errors"] += 1  # suite-level error (import/transform/setup)
    if rc == 0:
        return "survived", info
    if info["failures"] > 0:
        return "killed", info
    return "error", info


def run_suite(runner: str, cwd: Path) -> tuple[str, dict]:
    out = Path(tempfile.mkdtemp(prefix="docintel-mut-report-"))
    try:
        if runner == "py":
            junit = out / "junit.xml"
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--junitxml={junit}"]
            rc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).returncode
            return _classify_pytest(rc, junit)
        report = out / "vitest.json"
        vitest = cwd / "frontend" / "node_modules" / ".bin" / "vitest"
        rc = subprocess.run([str(vitest), "run", "--reporter=json", f"--outputFile={report}"],
                            cwd=cwd / "frontend", capture_output=True, text=True).returncode
        return _classify_vitest(rc, report)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-only", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    has_js = (ROOT / "frontend" / "node_modules" / ".bin" / "vitest").exists()
    results = []
    with tempfile.TemporaryDirectory(prefix="docintel-mut-") as tmp:
        base = Path(tmp) / "repo"
        copy_repo(base)
        runners = ("py",) + (("js",) if has_js and not args.backend_only else ())
        for r in runners:
            status, info = run_suite(r, base)
            if status != "survived":
                raise SystemExit(f"infrastructure failure: unmutated {r} suite did not pass cleanly: {info}")
        for mid, runner, rel, old, new, desc, expect in MUTANTS:
            entry = {"id": mid, "runner": runner, "file": rel, "description": desc, "expect": expect}
            if runner not in runners:
                entry["result"] = "not_run"
                entry["reason"] = ("disabled by --backend-only" if args.backend_only
                                   else "frontend tooling unavailable")
            else:
                path = base / rel
                original = path.read_text() if path.exists() else ""
                n = original.count(old)
                if n != 1:
                    entry["result"] = "obsolete"
                    entry["reason"] = f"target occurs {n} times in {rel} (must be exactly 1)"
                else:
                    path.write_text(original.replace(old, new))
                    t = time.perf_counter()
                    try:
                        status, info = run_suite(runner, base)
                    finally:
                        path.write_text(original)
                    entry.update(result=status, seconds=round(time.perf_counter() - t, 2),
                                 exit_code=info["exit_code"], failures=info["failures"],
                                 errors=info["errors"], failed_tests=info["failed_tests"][:5])
            results.append(entry)
            print(f"{mid}  {entry['result']:9s} {rel:38s} {desc}"
                  + (f"  [{entry.get('reason')}]" if entry.get("reason") else ""))
    counts = {k: sum(r["result"] == k for r in results)
              for k in ("killed", "survived", "error", "obsolete", "not_run")}
    summary = {"defined": len(results), **counts,
               "detection_rate_of_executed": f"{counts['killed']}/{counts['killed'] + counts['survived'] + counts['error']}"}
    print("\n" + json.dumps(summary))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"summary": summary, "mutants": results}, indent=2) + "\n")
    return 0 if counts["killed"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
