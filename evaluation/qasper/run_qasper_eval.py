#!/usr/bin/env python3
"""External retrieval benchmark on AllenAI QASPER (CC-BY-4.0), offline, local BGE-small (ONNX).

Documents: the dataset's own paper text (section names + paragraphs). NOT original PDFs and
NOT PDF-layout extraction (arxiv.org is not reachable here; see FROZEN settings).
Labels: QASPER's own annotator evidence paragraphs, unchanged. No labels are generated.

  --freeze      seeded paper/question selection + exclusion log + settings -> frozen.json
  --split dev   harness sanity on validation-split papers (settings are already fixed)
  --split test  final evaluation on test-split papers (paper-level separation by split)

Chunkers (same per-word BGE token counter, same budget T):
  fixed      backend.structure_chunker._windows over the whole paper: T tokens, overlap T/8
  structure  backend.structure_chunker.chunk_sections over (section, paragraphs): T tokens
Retrieval is per paper (QASPER setting): rank that paper's chunks for its question.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from backend import structure_chunker  # noqa: E402

_spec = importlib.util.spec_from_file_location("rre", ROOT / "evaluation" / "run_retrieval_eval.py")
rre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rre)

DATASET = ("allenai/qasper", "06806e4608976fc2fac0a090ac425d5b2b29caf4")  # parquet conversion commit
FROZEN = HERE / "frozen.json"
SETTINGS = {
    "seed": 20261125, "dev_min_questions": 40, "test_min_questions": 120,
    "chunk_budgets_T": [128, 256], "primary_T": 256, "overlap": "T // 8", "top_k": 4,
    "context_token_budgets": [512, 1024], "primary_endpoint": "evidence recall at 1024-token budget, dense, T=256",
    "evidence_match": "an evidence paragraph is retrieved if >= 50% of its lowercased 8-word shingles occur in the "
                      "retrieved chunks (paragraphs < 8 words: whole-paragraph containment)",
    "chunk_relevant": "a chunk is relevant if it contains >= 50% of the shingles of some evidence paragraph, or >= 50% "
                      "of its own shingles are evidence shingles",
    "annotators": "per question, recall/MRR = max over eligible annotations (alternative valid evidence sets); "
                  "precision uses the union of eligible annotations' evidence",
    "exclusions": "drop evidence strings starting with 'FLOAT SELECTED' (tables/figures); drop annotations that are "
                  "unanswerable, have no remaining evidence, or whose evidence does not exactly match (whitespace-"
                  "normalised) a paper paragraph; drop questions with no eligible annotation",
    "bootstrap": {"reps": 10000, "cluster": "paper", "ci": 0.95, "seed": 7},
    "decision": {"non_inferiority_margin": 0.05,
                 "rule": "structure - fixed on the primary endpoint: 'better' if CI lower bound > 0; "
                         "'non-inferior' if lower bound > -0.05; otherwise 'not shown non-inferior'"},
    "secondary": ["T=128", "BM25(k1=1.5,b=0.75)", "hybrid RRF k=10", "cross-encoder rerank of dense top-20"],
    "models": rre.MODELS,
}
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def load_split(split):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET[0], f"qasper/{split}/0000.parquet", repo_type="dataset", revision=DATASET[1])
    return pq.read_table(path).to_pylist()


def eligible_questions(paper, log):
    paras = {norm(p) for ps in paper["full_text"]["paragraphs"] for p in ps if norm(p)}
    out = []
    qa = paper["qas"]
    for i, qid in enumerate(qa["question_id"]):
        anns = []
        for a in qa["answers"][i]["answer"]:
            if a["unanswerable"]:
                log["annotation_unanswerable"] += 1
                continue
            ev = [norm(e) for e in a["evidence"] if norm(e) and not e.startswith("FLOAT SELECTED")]
            if not ev:
                log["annotation_no_text_evidence"] += 1
                continue
            if not all(e in paras for e in ev):
                log["annotation_evidence_unmapped"] += 1
                continue
            anns.append({"evidence": ev, "type": answer_type(a)})
        if anns:
            out.append({"question_id": qid, "question": qa["question"][i], "annotations": anns})
        else:
            log["question_excluded"] += 1
    return out


def answer_type(a):
    if a["extractive_spans"]:
        return "extractive"
    if a["yes_no"] is not None:
        return "yes_no"
    return "abstractive" if a["free_form_answer"] else "other"


def freeze():
    rng = random.Random(SETTINGS["seed"])
    frozen = {"dataset": DATASET, "settings": SETTINGS, "documents_are": "dataset-provided paper text, not PDFs"}
    for name, split, need in (("dev", "validation", SETTINGS["dev_min_questions"]), ("test", "test", SETTINGS["test_min_questions"])):
        papers = load_split(split)
        rng.shuffle(papers)
        log, chosen, nq = Counter(), [], 0
        for p in papers:
            if nq >= need:
                break
            qs = eligible_questions(p, log)
            if qs:
                chosen.append({"paper_id": p["id"], "questions": qs})
                nq += len(qs)
        types = Counter(t for c in chosen for q in c["questions"] for t in {a["type"] for a in q["annotations"]})
        frozen[name] = {"split": split, "papers": chosen, "n_papers": len(chosen), "n_questions": nq,
                        "exclusion_log": dict(log), "answer_types_among_eligible": dict(types)}
        print(name, len(chosen), "papers", nq, "questions", dict(log), dict(types))
    FROZEN.write_text(json.dumps(frozen, indent=1) + "\n")
    print("sha256", hashlib.sha256(FROZEN.read_bytes()).hexdigest())


# ----------------------------------------------------------------------------- evaluation
def shingles(text, n=8):
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


class Paper:
    def __init__(self, row, tok):
        ft = row["full_text"]
        self.sections = [(norm(h.replace(":::", " - ")), [norm(p) for p in ps if norm(p)])
                         for h, ps in zip(ft["section_name"], ft["paragraphs"])]
        self.words = [w for h, ps in self.sections for w in (h.split() + " ".join(ps).split())]
        self.tok = tok

    def chunks(self, kind, T):
        count = self.tok.n
        if kind == "fixed":
            return structure_chunker._windows(self.words, T, T // 8, count)
        return structure_chunker.chunk_sections(self.sections, T, count)


def covered(ev, sh):
    s = shingles(ev)
    return len(s & sh) / len(s)


def score_question(q, ranked_chunks, chunk_tokens, budgets, K):
    ranked_sh = [shingles(c) for c in ranked_chunks]
    union_ev = {e for a in q["annotations"] for e in a["evidence"]}
    ev_sh = set().union(*(shingles(e) for e in union_ev))

    def relevant(i):
        own = ranked_sh[i]
        return any(covered(e, own) >= 0.5 for e in union_ev) or len(own & ev_sh) / len(own) >= 0.5

    def recall_with(sel):
        sh = set().union(*(ranked_sh[i] for i in sel)) if sel else set()
        return max(sum(covered(e, sh) >= 0.5 for e in a["evidence"]) / len(a["evidence"]) for a in q["annotations"])

    rel = [relevant(i) for i in range(min(10, len(ranked_chunks)))]
    first = next((r + 1 for r, x in enumerate(rel) if x), None)
    out = {"recall@4": recall_with(list(range(min(K, len(ranked_chunks))))),
           "precision@4": sum(rel[:K]) / K, "rr@10": 1 / first if first else 0.0,
           "context_tokens@4": sum(chunk_tokens[:K])}
    for B in budgets:
        sel, used = [], 0
        for i, t in enumerate(chunk_tokens):
            if used + t > B:
                break
            sel.append(i)
            used += t
        out[f"recall@{B}tok"] = recall_with(sel)
        out[f"chunks@{B}tok"] = len(sel)
    return out


def paired_bootstrap(rows_a, rows_b, metric, reps, seed):
    by_paper = {}
    for a, b in zip(rows_a, rows_b):
        assert a["question_id"] == b["question_id"]
        by_paper.setdefault(a["paper_id"], []).append(b[metric] - a[metric])
    papers = list(by_paper)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        diffs = [d for p in rng.choice(papers, len(papers)) for d in by_paper[p]]
        means.append(statistics.fmean(diffs))
    point = statistics.fmean(d for ds in by_paper.values() for d in ds)
    return {"mean_diff": round(point, 4), "ci95": [round(float(np.percentile(means, 2.5)), 4),
                                                    round(float(np.percentile(means, 97.5)), 4)],
            "papers": len(papers), "questions": sum(map(len, by_paper.values()))}


def run(split_name):
    frozen = json.loads(FROZEN.read_text())
    part = frozen[split_name]
    rows_by_id = {r["id"]: r for r in load_split(part["split"])}
    dense = rre.Dense()
    ce = rre.CrossEncoder()
    from tokenizers import Tokenizer
    raw = Tokenizer.from_str(dense.m.tok.to_str())
    raw.no_truncation()
    raw.no_padding()
    tok = type("T", (), {"n": staticmethod(lambda t: len(raw.encode(t, add_special_tokens=False).ids))})
    S = frozen["settings"]
    configs = [(kind, T, ret) for T in S["chunk_budgets_T"] for kind in ("fixed", "structure") for ret in ("dense",)]
    configs += [(kind, S["primary_T"], ret) for kind in ("fixed", "structure") for ret in ("bm25", "hybrid", "rerank")]
    results = {f"{k}|T={T}|{r}": [] for k, T, r in configs}
    stats = {key: {"lat": [], "n_chunks": 0, "chunk_tokens": [], "bge_truncated": 0} for key in results}
    for paper in part["papers"]:
        P = Paper(rows_by_id[paper["paper_id"]], tok)
        cache = {}
        for kind, T, ret in configs:
            key = f"{kind}|T={T}|{ret}"
            if (kind, T) not in cache:
                ch = P.chunks(kind, T)
                cache[(kind, T)] = (ch, dense.encode(ch), [tok.n(c) for c in ch], rre.BM25(ch, 1.5, 0.75))
            ch, vec, ntok, bm = cache[(kind, T)]
            st = stats[key]
            st["n_chunks"] += len(ch)
            st["chunk_tokens"] += ntok
            st["bge_truncated"] += sum(n > 510 for n in ntok)
            for q in paper["questions"]:
                t0 = time.perf_counter()
                d = list(np.argsort(-(vec @ dense.encode([q["question"]])[0]), kind="stable"))
                if ret == "dense":
                    order = d
                elif ret == "bm25":
                    order = list(np.argsort(-bm.scores(q["question"]), kind="stable"))
                elif ret == "hybrid":
                    s = bm.scores(q["question"])
                    order = rre.rrf([d, [i for i in np.argsort(-s, kind="stable") if s[i] > 0]], 10)
                else:
                    head = d[:20]
                    sc = ce.score(q["question"], [ch[i] for i in head])
                    order = [head[j] for j in np.argsort(-sc, kind="stable")] + d[20:]
                st["lat"].append((time.perf_counter() - t0) * 1000)
                m = score_question(q, [ch[i] for i in order], [ntok[i] for i in order], S["context_token_budgets"], S["top_k"])
                results[key].append({"paper_id": paper["paper_id"], "question_id": q["question_id"], **m})
    metrics = ["recall@4", "precision@4", "rr@10", "context_tokens@4"] + [f"recall@{B}tok" for B in S["context_token_budgets"]]
    summary = {}
    for key, rows in results.items():
        st = stats[key]
        xs = sorted(st["lat"])
        summary[key] = {**{m: round(statistics.fmean(r[m] for r in rows), 4) for m in metrics},
                        "n_questions": len(rows), "n_chunks": st["n_chunks"],
                        "mean_chunk_tokens": round(statistics.fmean(st["chunk_tokens"]), 1),
                        "bge_truncated_chunks": st["bge_truncated"],
                        "latency_p50_ms": round(xs[math.ceil(0.5 * len(xs)) - 1], 2),
                        "latency_p95_ms": round(xs[math.ceil(0.95 * len(xs)) - 1], 2)}
    b = S["bootstrap"]
    comparisons = {}
    for T in S["chunk_budgets_T"]:
        for m in metrics:
            comparisons[f"structure-fixed|T={T}|dense|{m}"] = paired_bootstrap(
                results[f"fixed|T={T}|dense"], results[f"structure|T={T}|dense"], m, b["reps"], b["seed"])
    prim = comparisons[f"structure-fixed|T={S['primary_T']}|dense|recall@1024tok"]
    lo = prim["ci95"][0]
    margin = S["decision"]["non_inferiority_margin"]
    verdict = "better" if lo > 0 else ("non-inferior" if lo > -margin else "not shown non-inferior")
    out = {"split": split_name, "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
           "summary": summary, "paired_comparisons": comparisons, "primary": {**prim, "verdict": verdict},
           "per_query": results}
    (HERE / f"results_{split_name}.json").write_text(json.dumps(out, indent=1) + "\n")
    for key, s in summary.items():
        print(f"{key:28s} R@4 {s['recall@4']:.3f} P@4 {s['precision@4']:.3f} MRR {s['rr@10']:.3f} tok@4 {s['context_tokens@4']:6.1f} "
              f"R@512t {s['recall@512tok']:.3f} R@1024t {s['recall@1024tok']:.3f} p50 {s['latency_p50_ms']}ms trunc {s['bge_truncated_chunks']}")
    print("PRIMARY", out["primary"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"])
    args = ap.parse_args()
    if args.freeze:
        if FROZEN.exists():
            sys.exit("frozen.json already exists; refusing to re-sample")
        freeze()
    else:
        run(args.split)
