#!/usr/bin/env python3
"""Context-efficiency evaluation: adaptive / score-based / de-duplicated / token-budgeted
selection of passages from the dense top-4, compared with fixed top-4 dense retrieval.

Ranking: BAAI/bge-small-en-v1.5 (local ONNX, cosine; the dev-best dense config). All methods
select a subset of the same top-4 in rank order, so differences are purely context selection.

Per query (then averaged; retention is micro-averaged):
  context_tokens = bge WordPiece tokens of the selected passages (a proxy: Gemini tokenisation differs)
  precision      = |relevant ∩ selected| / |selected|
  recall         = |relevant ∩ selected| / |relevant|
  rr             = 1 / rank of the first relevant passage within the selected set (0 if none)
  retention      = Σ|relevant ∩ selected| / Σ|relevant ∩ fixed top-4|   (relevant-evidence retention)
  latency_ms     = query encoding + search + selection
Custom IR metrics, not RAGAS.

Pre-registered rules (written to results/context_selection.json before the unseen set is run):
  DEV SELECTION   among non-baseline configs with dev retention >= 0.95 and dev recall >=
                  baseline recall - 0.02, choose the lowest mean context tokens (ties: higher
                  precision, then fewer active mechanisms).
  GENERALISES     on the frozen unseen set: token reduction >= 25 % vs fixed top-4, retention
                  >= 0.95 and recall drop <= 0.05.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
RESULTS = EVAL / "results"
_spec = importlib.util.spec_from_file_location("rre", EVAL / "run_retrieval_eval.py")
rre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rre)

BASE_K = 4
RULES = {"dev_min_retention": 0.95, "dev_max_recall_drop": 0.02,
         "gen_min_token_reduction": 0.25, "gen_min_retention": 0.95, "gen_max_recall_drop": 0.05}


def select_passages(cfg, scores, cand, pvec, ntok):
    """cand: top-4 passage indices in rank order; scores: their cosine similarities."""
    keep = list(range(len(cand)))
    if cfg.get("k"):
        keep = keep[:cfg["k"]]
    if cfg.get("abs") is not None:
        keep = [j for j in keep if scores[j] >= cfg["abs"]] or [0]
    if cfg.get("rel") is not None:
        keep = [j for j in keep if scores[j] >= scores[0] - cfg["rel"]] or [0]
    if cfg.get("gap"):
        drops = [scores[j] - scores[j + 1] for j in range(len(cand) - 1)]
        cut = int(np.argmax(drops)) + 1
        keep = [j for j in keep if j < cut] or [0]
    if cfg.get("dedup") is not None:
        kept = []
        for j in keep:
            if all(float(pvec[cand[j]] @ pvec[cand[i]]) <= cfg["dedup"] for i in kept):
                kept.append(j)
        keep = kept
    if cfg.get("budget"):
        out, used = [], 0
        for j in keep:
            if out and used + ntok[cand[j]] > cfg["budget"]:
                break
            out.append(j)
            used += ntok[cand[j]]
        keep = out
    return [cand[j] for j in keep]


def cfg_name(cfg):
    if not cfg:
        return "fixed-top4 (baseline)"
    return " + ".join(f"{k}={v}" for k, v in cfg.items())


def grid():
    cfgs = [{}] + [{"k": k} for k in (1, 2, 3)] + [{"gap": True}]
    cfgs += [{"abs": t} for t in (0.60, 0.65, 0.70, 0.75, 0.80)]
    rels = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    cfgs += [{"rel": d} for d in rels]
    cfgs += [{"dedup": r} for r in (0.85, 0.90, 0.95)]
    cfgs += [{"budget": b} for b in (64, 96, 128, 160)]
    cfgs += [{"rel": d, "dedup": r} for d, r in itertools.product(rels, (0.90, 0.95))]
    cfgs += [{"rel": d, "budget": b} for d, b in itertools.product(rels, (96, 128))]
    return cfgs


class Context:
    def __init__(self):
        passages, _ = rre.load()
        self.passages = passages
        self.ids = [p["passage_id"] for p in passages]
        self.dense = rre.Dense()
        self.pvec = self.dense.encode([p["text"] for p in passages])
        tok = self.dense.m.tok
        self.ntok = [len(tok.encode(p["text"], add_special_tokens=False).ids) for p in passages]

    def run(self, cfg, questions):
        rows, lat = [], []
        for q in questions[:2]:  # warm-up
            self._one(cfg, q)
        for q in questions:
            t = time.perf_counter()
            sel, top4 = self._one(cfg, q)
            lat.append((time.perf_counter() - t) * 1000)
            rel = set(q["relevant_passage_ids"])
            sel_ids = [self.ids[i] for i in sel]
            hits = len(rel & set(sel_ids))
            first = next((r + 1 for r, p in enumerate(sel_ids) if p in rel), None)
            rows.append({"qid": q["qid"], "selected": sel_ids, "top4": [self.ids[i] for i in top4],
                         "relevant": sorted(rel), "n_selected": len(sel),
                         "context_tokens": sum(self.ntok[i] for i in sel),
                         "precision": hits / len(sel), "recall": hits / len(rel), "rr": 1 / first if first else 0.0,
                         "relevant_selected": hits, "relevant_in_top4": len(rel & {self.ids[i] for i in top4}),
                         "latency_ms": round(lat[-1], 3)})
        xs = sorted(lat)
        s = {m: round(statistics.fmean(r[m] for r in rows), 4)
             for m in ("context_tokens", "n_selected", "precision", "recall", "rr")}
        s["retention"] = round(sum(r["relevant_selected"] for r in rows) / max(1, sum(r["relevant_in_top4"] for r in rows)), 4)
        s["latency_p50_ms"] = round(xs[math.ceil(0.5 * len(xs)) - 1], 3)
        s["latency_p95_ms"] = round(xs[math.ceil(0.95 * len(xs)) - 1], 3)
        s["n"] = len(rows)
        return {"name": cfg_name(cfg), "config": cfg, "summary": s, "per_query": rows}

    def _one(self, cfg, q):
        qv = self.dense.encode([q["question"]])[0]
        sims = self.pvec @ qv
        top4 = list(np.argsort(-sims, kind="stable")[:BASE_K])
        return select_passages(cfg, [float(sims[i]) for i in top4], top4, self.pvec, self.ntok), top4


def load_questions(split):
    return [json.loads(l) for l in (EVAL / f"questions/{split}.jsonl").read_text().splitlines()]


def verify_locks():
    rre.check_labels()
    lock = json.loads((EVAL / "UNSEEN.lock.json").read_text())
    for rel, digest in lock["sha256"].items():
        if hashlib.sha256((EVAL / rel).read_bytes()).hexdigest() != digest:
            sys.exit(f"frozen file changed: {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "unseen", "heldout_historical"], required=True)
    args = ap.parse_args()
    verify_locks()
    ctx = Context()
    RESULTS.mkdir(exist_ok=True)
    sel_path = RESULTS / "context_selection.json"
    if args.split == "dev":
        res = [ctx.run(c, load_questions("dev")) for c in grid()]
        base = res[0]["summary"]
        ok = [r for r in res[1:] if r["summary"]["retention"] >= RULES["dev_min_retention"]
              and r["summary"]["recall"] >= base["recall"] - RULES["dev_max_recall_drop"]]
        chosen = min(ok, key=lambda r: (r["summary"]["context_tokens"], -r["summary"]["precision"], len(r["config"]))) if ok else None
        (sel_path).write_text(json.dumps({"rules": RULES, "selected": chosen and chosen["config"],
                                          "selected_name": chosen and chosen["name"],
                                          "dev_baseline": base, "dev_selected": chosen and chosen["summary"]}, indent=1) + "\n")
        out = {"split": "dev", "results": res}
        shown = [res[0]] + sorted(ok, key=lambda r: r["summary"]["context_tokens"])[:6]
    else:
        if not sel_path.exists():
            sys.exit("run --split dev first")
        sel = json.loads(sel_path.read_text())
        if not sel["selected"]:
            sys.exit("no configuration qualified on dev")
        split = "heldout" if args.split == "heldout_historical" else "unseen"
        res = [ctx.run({}, load_questions(split)), ctx.run(sel["selected"], load_questions(split))]
        b, s = res[0]["summary"], res[1]["summary"]
        verdict = {"token_reduction": round(1 - s["context_tokens"] / b["context_tokens"], 4),
                   "retention": s["retention"], "recall_drop": round(b["recall"] - s["recall"], 4)}
        verdict["generalises"] = (verdict["token_reduction"] >= RULES["gen_min_token_reduction"]
                                  and verdict["retention"] >= RULES["gen_min_retention"]
                                  and verdict["recall_drop"] <= RULES["gen_max_recall_drop"])
        out = {"split": args.split, "selection": sel, "verdict": verdict, "results": res}
        shown = res
        print("VERDICT", verdict)
    (RESULTS / f"context_{args.split}.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in shown:
        s = r["summary"]
        print(f"{r['name'][:34]:34s} tok {s['context_tokens']:6.1f} n {s['n_selected']:.2f} P {s['precision']:.3f} "
              f"R {s['recall']:.3f} MRR {s['rr']:.3f} ret {s['retention']:.3f} p50 {s['latency_p50_ms']:.2f}ms")
    if args.split == "dev":
        print("SELECTED:", chosen and chosen["name"])


if __name__ == "__main__":
    main()
