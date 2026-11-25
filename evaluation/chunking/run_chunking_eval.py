#!/usr/bin/env python3
"""Structure-aware + token-aware chunking vs fixed token windows vs the production chunker,
through real PDF extraction (PyMuPDF) and the local BAAI/bge-small-en-v1.5 embedder.

Corpus: the 9 frozen documents rendered into 4 multi-section PDFs, one text block per
paragraph (as real PDFs have). Production/fixed chunkers use backend.ingest.parse_pdf text;
the structure-aware chunker uses the same file's text *blocks* (get_text("blocks")).

Configurations
  A  production chunk_text (800 words / 100 overlap)  -- exceeds BGE's 512-token input, so its
     dense scores are degraded by LOCAL truncation; Gemini embedding accepts 2,048 tokens.
     Reported, flagged, NOT used for acceptance.
  B  fixed token windows of T BGE tokens, overlap T/8 (size-matched control)
  C  structure-aware: paragraphs packed within a section up to T tokens, oversize paragraphs
     split by token windows; optional section heading prefixed to each chunk
Fair-comparison controls: B and C share the T grid, stay within the BGE limit, use the same
embedder, query encoding, top-k and span-level evidence labels (independent of chunk bounds).

Metrics (k=4): evidence recall@4 = spans contained in some top-4 chunk / spans;
precision@4 = top-4 chunks containing >=1 span / 4; MRR@10 = 1/rank of first chunk containing a
span; context tokens = BGE tokens in the top-4; truncation = chunks > 510 content tokens (BGE
limit) and > 2,048 (Gemini limit, BGE tokens as proxy); query latency (encode+search) and
index time (chunk+embed). Custom IR metrics, not RAGAS.

PRE-REGISTERED ACCEPTANCE (test split, C* vs B*, each selected on dev by recall, MRR, -tokens):
  recall_C >= recall_B - 0.02  AND  no BGE truncation in C  AND
  (MRR_C >= MRR_B + 0.05  OR  recall_C >= recall_B + 0.05  OR  tokens_C <= 0.80 * tokens_B)
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEMINI_API_KEY", "offline-not-used")
os.environ.setdefault("GROQ_API_KEY", "offline-not-used")
mock.patch("google.genai.Client").start()
from backend import ingest  # noqa: E402  production parse_pdf / chunk_text
import fitz  # noqa: E402

_spec = importlib.util.spec_from_file_location("rre", EVAL / "run_retrieval_eval.py")
rre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rre)

GROUPS = {"handbook.pdf": ["northwind_travel_policy_2024", "northwind_travel_policy_2022", "northwind_it_security_policy"],
          "vendor_contracts.pdf": ["contoso_cloud_msa", "fabrikam_logistics_saas"],
          "patient_R1047_lab_reports.pdf": ["riverside_lab_report_jan2025", "riverside_lab_report_jul2025"],
          "biology_study_notes.pdf": ["biology_cellular_respiration_notes", "biology_photosynthesis_notes"]}
K, BGE_LIMIT, GEMINI_LIMIT = 4, 510, 2048
ACCEPT = {"max_recall_drop": 0.02, "min_mrr_gain": 0.05, "min_recall_gain": 0.05, "max_token_ratio": 0.80}
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def build_pdfs(folder: Path):
    docs = {}
    for line in (EVAL / "corpus/passages.jsonl").read_text().splitlines():
        p = json.loads(line)
        docs.setdefault(p["doc_id"], {"title": p["title"], "paras": []})["paras"].append(p["text"])
    for fname, ids in GROUPS.items():
        pdf = fitz.open()
        page, y = pdf.new_page(), 40
        for doc_id in ids:
            for text, size in [(docs[doc_id]["title"], 12)] + [(t, 9) for t in docs[doc_id]["paras"]]:
                for _ in range(2):
                    rect = fitz.Rect(40, y, page.rect.width - 40, page.rect.height - 40)
                    spare = page.insert_textbox(rect, text, fontsize=size)
                    if spare >= 0:
                        y += rect.height - spare + 8
                        break
                    page, y = pdf.new_page(), 40
                else:
                    raise RuntimeError("paragraph does not fit on a page")
        pdf.save(folder / fname)


class Tok:
    """Counts BGE WordPiece tokens WITHOUT truncation (the embedding tokenizer truncates at 512,
    which would cap counts; a first run had this bug and understated configuration A's tokens)."""
    def __init__(self, tokenizer):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_str(tokenizer.to_str())
        self.t.no_truncation()
        self.t.no_padding()

    def n(self, text):
        return len(self.t.encode(text, add_special_tokens=False).ids)


def windows(words, tok, T, overlap):
    wt = [max(1, tok.n(w)) for w in words]
    out, i = [], 0
    while i < len(words):
        j, used = i, 0
        while j < len(words) and (used + wt[j] <= T or j == i):
            used += wt[j]
            j += 1
        out.append(" ".join(words[i:j]))
        if j >= len(words):
            break
        back, k = 0, j
        while k > i + 1 and back + wt[k - 1] <= overlap:
            k -= 1
            back += wt[k]
        i = k
    return out


def chunk_production(path, tok, **_):
    return ingest.chunk_text(ingest.parse_pdf(str(path)))


def chunk_fixed(path, tok, T, **_):
    return windows(ingest.parse_pdf(str(path)).split(), tok, T, T // 8)


def chunk_structure(path, tok, T, header=False, **_):
    blocks = [norm(b[4]) for page in fitz.open(path) for b in page.get_text("blocks") if b[6] == 0 and norm(b[4])]
    sections, cur = [], None
    for b in blocks:
        if len(b.split()) <= 12 and not b.endswith("."):
            cur = {"head": b, "paras": []}
            sections.append(cur)
        else:
            if cur is None:
                cur = {"head": "", "paras": []}
                sections.append(cur)
            cur["paras"].append(b)
    chunks = []
    for s in sections:
        prefix = (s["head"] + ". ") if header and s["head"] else ""
        budget = T - tok.n(prefix)
        pieces = []
        for p in s["paras"]:
            pieces += [p] if tok.n(p) <= budget else windows(p.split(), tok, budget, budget // 8)
        if not header and s["head"]:
            pieces = [s["head"] + ". " + pieces[0]] + pieces[1:] if pieces else [s["head"]]
        cur_text = ""
        for p in pieces:
            cand = (cur_text + " " + p).strip()
            if cur_text and tok.n(prefix + cand) > T:
                chunks.append(prefix + cur_text)
                cur_text = p
            else:
                cur_text = cand
        if cur_text:
            chunks.append(prefix + cur_text)
    return chunks


CHUNKERS = {"A": chunk_production, "B": chunk_fixed, "C": chunk_structure}


def cfg_name(c):
    if c["family"] == "A":
        return "A production (800/100 words)"
    if c["family"] == "B":
        return f"B fixed windows T={c['T']}"
    return f"C structure-aware T={c['T']}{' +heading' if c.get('header') else ''}"


class Experiment:
    def __init__(self):
        self.dense = rre.Dense()
        self.tok = Tok(self.dense.m.tok)
        self.dir = Path(tempfile.mkdtemp(prefix="docintel-chunk-"))
        build_pdfs(self.dir)
        self.pdfs = sorted(self.dir.glob("*.pdf"))

    def index(self, cfg):
        t = time.perf_counter()
        chunks = [norm(c) for p in self.pdfs for c in CHUNKERS[cfg["family"]](p, self.tok, **cfg)]
        vecs = self.dense.encode(chunks)
        return chunks, vecs, (time.perf_counter() - t) * 1000

    def run(self, cfg, questions):
        chunks, vecs, index_ms = self.index(cfg)
        ntok = [self.tok.n(c) for c in chunks]
        rows, lat = [], []
        for q in questions:
            t = time.perf_counter()
            sims = vecs @ self.dense.encode([q["question"]])[0]
            order = list(np.argsort(-sims, kind="stable")[:10])
            lat.append((time.perf_counter() - t) * 1000)
            spans = [norm(s) for s in q["evidence_spans"]]
            has = [any(s in chunks[i] for s in spans) for i in order]
            found = [s for s in spans if any(s in chunks[i] for i in order[:K])]
            first = next((r + 1 for r, h in enumerate(has) if h), None)
            rows.append({"qid": q["qid"], "recall@4": len(found) / len(spans), "precision@4": sum(has[:K]) / K,
                         "rr@10": 1 / first if first else 0.0, "context_tokens": sum(ntok[i] for i in order[:K]),
                         "spans_missing_from_top4": [s for s in spans if s not in found],
                         "latency_ms": round(lat[-1], 3)})
        all_spans = {norm(s) for q in questions for s in q["evidence_spans"]}
        xs = sorted(lat)
        summ = {m: round(statistics.fmean(r[m] for r in rows), 4) for m in ("recall@4", "precision@4", "rr@10", "context_tokens")}
        summ.update(n_questions=len(rows), n_chunks=len(chunks), mean_chunk_tokens=round(statistics.fmean(ntok), 1),
                    max_chunk_tokens=max(ntok),
                    truncated_by_bge_512=round(sum(n > BGE_LIMIT for n in ntok) / len(ntok), 4),
                    over_gemini_2048_proxy=round(sum(n > GEMINI_LIMIT for n in ntok) / len(ntok), 4),
                    spans_split_by_chunking=round(sum(not any(s in c for c in chunks) for s in all_spans) / len(all_spans), 4),
                    query_latency_p50_ms=round(xs[math.ceil(0.5 * len(xs)) - 1], 3),
                    query_latency_p95_ms=round(xs[math.ceil(0.95 * len(xs)) - 1], 3), index_ms=round(index_ms, 1))
        return {"name": cfg_name(cfg), "config": cfg, "summary": summ, "per_query": rows}


def load(name):
    lock = json.loads((HERE / "LOCK.json").read_text())
    for f, d in lock["sha256"].items():
        if hashlib.sha256((HERE / f).read_bytes()).hexdigest() != d:
            sys.exit(f"frozen file changed: {f}")
    return [json.loads(l) for l in (HERE / name).read_text().splitlines()]


def best(results):
    return max(results, key=lambda r: (r["summary"]["recall@4"], r["summary"]["rr@10"], -r["summary"]["context_tokens"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    exp = Experiment()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    sel_path = out_dir / "selection.json"
    if args.split == "dev":
        qs = load("dev_evidence.jsonl")
        grid = [{"family": "A"}] + [{"family": "B", "T": T} for T in (128, 192, 256, 384)] + \
               [{"family": "C", "T": T, "header": h} for T in (128, 192, 256, 384) for h in (False, True)]
        res = [exp.run(c, qs) for c in grid]
        sel = {"acceptance": ACCEPT, "B": best([r for r in res if r["config"]["family"] == "B"])["config"],
               "C": best([r for r in res if r["config"]["family"] == "C"])["config"]}
        sel_path.write_text(json.dumps(sel, indent=1) + "\n")
        out = {"split": "dev", "selection": sel, "results": res}
    else:
        sel = json.loads(sel_path.read_text())
        qs = load("test_questions.jsonl")
        res = [exp.run(c, qs) for c in ({"family": "A"}, sel["B"], sel["C"], sel["supplementary_size_matched_B"])]
        b, c = res[1]["summary"], res[2]["summary"]
        v = {"recall_delta": round(c["recall@4"] - b["recall@4"], 4), "mrr_delta": round(c["rr@10"] - b["rr@10"], 4),
             "token_ratio": round(c["context_tokens"] / b["context_tokens"], 4), "c_truncation": c["truncated_by_bge_512"]}
        v["accepted"] = (v["recall_delta"] >= -ACCEPT["max_recall_drop"] and v["c_truncation"] == 0 and
                         (v["mrr_delta"] >= ACCEPT["min_mrr_gain"] or v["recall_delta"] >= ACCEPT["min_recall_gain"]
                          or v["token_ratio"] <= ACCEPT["max_token_ratio"]))
        out = {"split": "test", "selection": sel, "verdict": v, "results": res}
        print("VERDICT", v)
    (out_dir / f"{args.split}.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in res:
        s = r["summary"]
        print(f"{r['name']:34s} R@4 {s['recall@4']:.3f} P@4 {s['precision@4']:.3f} MRR {s['rr@10']:.3f} tok {s['context_tokens']:6.1f} "
              f"chunks {s['n_chunks']:3d} mean {s['mean_chunk_tokens']:5.1f} max {s['max_chunk_tokens']:4d} trunc {s['truncated_by_bge_512']:.2f} "
              f"split {s['spans_split_by_chunking']:.2f} q50 {s['query_latency_p50_ms']:.2f}ms idx {s['index_ms']:.0f}ms")


if __name__ == "__main__":
    main()
