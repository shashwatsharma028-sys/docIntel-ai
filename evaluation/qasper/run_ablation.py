#!/usr/bin/env python3
"""Controlled retrieval ablation on QASPER (offline, local BGE): A dense, B dense+BM25 RRF,
C B+cross-encoder rerank of the top-20 fused candidates, REF dense top-20 reranked.
Everything else fixed per ablation_criteria.json (structure-aware T=256, top-4, 1,024-token budget)."""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qasper_eval as rq  # noqa: E402

CRIT = json.loads((HERE / "ablation_criteria.json").read_text())
FIX = CRIT["fixed"]
METRICS = ["recall@4", "precision@4", "rr@10", "context_tokens@4", "recall@512tok", "recall@1024tok"]


def main(split):
    frozen = json.loads(rq.FROZEN.read_text())
    part = frozen[split]
    rows = {r["id"]: r for r in rq.load_split(part["split"])}
    dense, ce = rq.rre.Dense(), rq.rre.CrossEncoder()
    from tokenizers import Tokenizer
    raw = Tokenizer.from_str(dense.m.tok.to_str())
    raw.no_truncation()
    raw.no_padding()
    tok = type("T", (), {"n": staticmethod(lambda t: len(raw.encode(t, add_special_tokens=False).ids))})
    pool, k1, b, rk = FIX["candidate_pool"], FIX["bm25"]["k1"], FIX["bm25"]["b"], FIX["rrf_k"]
    res = {c: [] for c in ("A", "B", "C", "REF")}
    lat = {c: [] for c in res}
    for paper in part["papers"]:
        P = rq.Paper(rows[paper["paper_id"]], tok)
        ch = P.chunks("structure", 256)
        vec, ntok, bm = dense.encode(ch), [tok.n(c) for c in ch], rq.rre.BM25(ch, k1, b)
        for q in paper["questions"]:
            t0 = time.perf_counter()
            d = list(np.argsort(-(vec @ dense.encode([q["question"]])[0]), kind="stable"))
            t_dense = time.perf_counter() - t0
            t1 = time.perf_counter()
            s = bm.scores(q["question"])
            fused = rq.rre.rrf([d, [i for i in np.argsort(-s, kind="stable") if s[i] > 0]], rk)
            fused += [i for i in d if i not in set(fused)]  # keep every chunk rankable
            t_fuse = time.perf_counter() - t1

            def rerank(order):
                t = time.perf_counter()
                head = order[:pool]
                sc = ce.score(q["question"], [ch[i] for i in head])
                return [head[j] for j in np.argsort(-sc, kind="stable")] + order[pool:], time.perf_counter() - t
            c_order, t_rc = rerank(fused)
            r_order, t_rr = rerank(d)
            for name, order, t in (("A", d, t_dense), ("B", fused, t_dense + t_fuse),
                                   ("C", c_order, t_dense + t_fuse + t_rc), ("REF", r_order, t_dense + t_rr)):
                m = rq.score_question(q, [ch[i] for i in order], [ntok[i] for i in order], [512, 1024], FIX["top_k"])
                res[name].append({"paper_id": paper["paper_id"], "question_id": q["question_id"], **m})
                lat[name].append(t * 1000)
    summary = {}
    for c, rs in res.items():
        xs = sorted(lat[c])
        summary[c] = {**{m: round(statistics.fmean(r[m] for r in rs), 4) for m in METRICS}, "n": len(rs),
                      "latency_p50_ms": round(xs[math.ceil(.5 * len(xs)) - 1], 2),
                      "latency_p95_ms": round(xs[math.ceil(.95 * len(xs)) - 1], 2)}
    steps = {"B-A": ("A", "B"), "C-B": ("B", "C"), "C-A": ("A", "C"), "REF-A": ("A", "REF"), "C-REF": ("REF", "C")}
    paired = {k: {m: rq.paired_bootstrap(res[a], res[b], m, 10000, 7) for m in ("recall@1024tok", "rr@10", "recall@4", "precision@4")}
              for k, (a, b) in steps.items()}
    out = {"split": split, "criteria": CRIT, "summary": summary, "paired": paired, "per_query": res}
    if split == "dev":
        sel = select(summary)
        out["selection"] = sel
        (HERE / "ablation_selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    (HERE / f"ablation_{split}.json").write_text(json.dumps(out, indent=1) + "\n")
    for c, s in summary.items():
        print(f"{c:4s} R@4 {s['recall@4']:.3f} P@4 {s['precision@4']:.3f} MRR {s['rr@10']:.3f} R@1024t {s['recall@1024tok']:.3f} "
              f"tok@4 {s['context_tokens@4']:.0f} p50 {s['latency_p50_ms']}ms p95 {s['latency_p95_ms']}ms")
    for k, v in paired.items():
        print(k, {m: (x["mean_diff"], x["ci95"]) for m, x in v.items()})
    if split == "dev":
        print("SELECTION", json.dumps(out["selection"]))


def advances(prev, cur):
    dp = cur["recall@1024tok"] - prev["recall@1024tok"]
    dm = cur["rr@10"] - prev["rr@10"]
    return dp >= 0.02 or (dm >= 0.03 and dp >= -0.02)


def select(summary):
    chain, last = ["A"], "A"
    decisions = {}
    if advances(summary["A"], summary["B"]):
        chain.append("B")
        last = "B"
    decisions["B_over_A"] = last == "B"
    c_base = "B" if last == "B" else "A"
    # C is defined as reranking the last advancing configuration's candidates (B if it advanced, else REF ~ A+rerank)
    c_name = "C" if c_base == "B" else "REF"
    decisions["rerank_over_" + c_base] = advances(summary[c_base], summary[c_name])
    if decisions["rerank_over_" + c_base]:
        chain.append(c_name)
        last = c_name
    return {"rule": CRIT["selection_on_dev"]["advance_rule"], "decisions": decisions, "chain": chain, "selected": last}


if __name__ == "__main__":
    main(sys.argv[1])
