#!/usr/bin/env python3
"""Targeted mutation testing for the Phase 0 baseline suite.

Each mutant is a single, deliberate defect injected into a THROWAWAY COPY of the
repository (the working tree is never modified). The relevant suite is run
against the copy; a mutant is "killed" when the suite fails. Surviving mutants
show assertions that do not detect that defect.

Mutants marked expect="xpass" deliberately FIX a documented known defect; the
suite must fail because a strict xfail unexpectedly passes -- proving that the
expected-failure tests detect behaviour changes in both directions.

usage:  python scripts/mutation_check.py [--backend-only] [--json OUT]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (id, runner, file, old, new, description, expect)   expect: "killed" | "xpass"
MUTANTS = [
    ("M01", "py", "backend/ingest.py", "CHUNK_OVERLAP = 100", "CHUNK_OVERLAP = 0", "remove chunk overlap", "killed"),
    ("M02", "py", "backend/ingest.py", "i += CHUNK_SIZE - CHUNK_OVERLAP", "i += CHUNK_SIZE - CHUNK_OVERLAP + 1", "off-by-one window step", "killed"),
    ("M03", "py", "backend/ingest.py", "len(c.strip()) > 50", "len(c.strip()) > 0", "drop short-chunk filter", "killed"),
    ("M04", "py", "backend/ingest.py", "words[i:i + CHUNK_SIZE]", "words[i:i + CHUNK_SIZE - 1]", "chunk loses last word", "killed"),
    ("M05", "py", "backend/ingest.py", "page.get_text() for page in doc", "page.get_text() for page in list(doc)[:1]", "extract first page only", "killed"),
    ("M06", "py", "backend/ingest.py", "faiss.IndexFlatL2(", "faiss.IndexFlatIP(", "wrong FAISS metric/index type", "killed"),
    ("M07", "py", "backend/ingest.py", "[(filename, c) for c", "[(c, filename) for c", "swap source/text in persisted chunks", "killed"),
    ("M08", "py", "backend/ingest.py", 'model="gemini-embedding-001"', 'model="text-embedding-004"', "wrong embedding model (ingest)", "killed"),
    ("M09", "py", "backend/retriever.py", 'chunks[i][0], "text": chunks[i][1]', 'chunks[i][1], "text": chunks[i][0]', "source mapping swapped", "killed"),
    ("M10", "py", "backend/retriever.py", "top_k: int = 4", "top_k: int = 3", "default top_k changed", "killed"),
    ("M11", "py", "backend/retriever.py", "for i in indices[0]", "for i in indices[0][::-1]", "ranking order reversed", "killed"),
    ("M12", "py", "backend/retriever.py", "if i < len(chunks)", "if 0 <= i < len(chunks)", "FIX D-01 (must trip strict xfail)", "xpass"),
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
    ("F01", "js", "frontend/src/App.jsx", "JSON.stringify({ question, mode })", "JSON.stringify({ question })", "chat request drops mode", "killed"),
    ("F02", "js", "frontend/src/App.jsx", "    setMode(newMode)\n    setMessages([])", "    setMode(newMode)", "mode switch keeps history", "killed"),
    ("F03", "js", "frontend/src/components/Message.jsx", 'high: "text-green-400"', 'high: "text-red-400"', "confidence colour mapping", "killed"),
    ("F04", "js", "frontend/src/components/Message.jsx", 'message.sources.length > 1 ? "s" : ""', '"s"', "source pluralisation", "killed"),
    ("F05", "js", "frontend/src/components/Chat.jsx", 'e.key === "Enter" && !e.shiftKey', 'e.key === "Enter"', "Shift+Enter sends", "killed"),
    ("F06", "js", "frontend/src/App.jsx", "setUploadedFiles((prev) => [...prev, file.name])", "setUploadedFiles((prev) => prev)", "uploaded file not listed", "killed"),
]


def copy_repo(dst: Path):
    ignore = shutil.ignore_patterns(".git", "node_modules", "__pycache__", ".pytest_cache", "coverage", ".coverage")
    shutil.copytree(ROOT, dst, ignore=ignore, symlinks=True)
    nm = ROOT / "frontend" / "node_modules"
    if nm.exists():
        os.symlink(nm.resolve(), dst / "frontend" / "node_modules")


def run_suite(runner: str, cwd: Path) -> int:
    if runner == "py":
        cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).returncode
    vitest = (cwd / "frontend" / "node_modules" / ".bin" / "vitest")
    return subprocess.run([str(vitest), "run"], cwd=cwd / "frontend", capture_output=True, text=True).returncode


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
        # sanity: unmutated suites must pass
        for r in ("py",) + (("js",) if has_js and not args.backend_only else ()):
            assert run_suite(r, base) == 0, f"unmutated {r} suite fails; aborting"
        for mid, runner, rel, old, new, desc, expect in MUTANTS:
            if expect == "skip" or (runner == "js" and (args.backend_only or not has_js)):
                continue
            path = base / rel
            original = path.read_text()
            if original.count(old) != 1:
                raise SystemExit(f"{mid}: target must occur exactly once in {rel} (found {original.count(old)})")
            path.write_text(original.replace(old, new))
            try:
                rc = run_suite(runner, base)
            finally:
                path.write_text(original)
            detected = rc != 0
            results.append({"id": mid, "file": rel, "description": desc, "expect": expect,
                            "result": "killed" if detected else "SURVIVED"})
            print(f"{mid}  {'killed  ' if detected else 'SURVIVED'}  {rel:38s} {desc}")
    killed = sum(r["result"] == "killed" for r in results)
    print(f"\n{killed}/{len(results)} mutants detected")
    if args.json:
        Path(args.json).write_text(json.dumps({"detected": killed, "total": len(results), "mutants": results}, indent=2))
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
