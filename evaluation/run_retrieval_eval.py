#!/usr/bin/env python3
"""Phase 2 retrieval-quality evaluation on the frozen labels (evaluation/LABELS.lock.json).

Units: the 48 frozen pre-segmented passages (NOT production 800-word chunks; each corpus
document is < 200 words, so production chunking would yield one chunk per document).

Systems: dense (BAAI/bge-small-en-v1.5, ONNX, CLS pooling, cosine), BM25 (Okapi), hybrid
(reciprocal rank fusion of dense + BM25), cross-encoder rerank
(cross-encoder/ms-marco-MiniLM-L-6-v2, ONNX) of a first-stage top-N.

Metrics (k = 4, the production top_k), per query then macro-averaged:
  Precision@4 = |relevant ∩ top-4| / 4
  Recall@4    = |relevant ∩ top-4| / |relevant|
  MRR@10      = 1 / rank of the first relevant passage within the top-10 (0 if none)
  Latency     = wall time per query on CPU, including query encoding and reranking
These are standard IR metrics computed here; they are NOT RAGAS metrics.

Protocol: `--split dev` evaluates the grid and writes the pre-registered selection
(max dev MRR@10, then Recall@4, then lower p50 latency) to results/selection.json.
`--split heldout` evaluates only configurations fixed on dev; it refuses to run without
a selection file. Real pretrained models only; no API calls.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
RESULTS = EVAL / "results"
MODELS = {
    "embed": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    "rerank": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 4
THREADS = 4


def check_labels():
    lock = json.loads((EVAL / "LABELS.lock.json").read_text())
    for rel, digest in lock["sha256"].items():
        if hashlib.sha256((EVAL / rel).read_bytes()).hexdigest() != digest:
            sys.exit(f"frozen file changed: {rel}")


def load():
    passages = [json.loads(l) for l in (EVAL / "corpus/passages.jsonl").read_text().splitlines()]
    qs = {s: [json.loads(l) for l in (EVAL / f"questions/{s}.jsonl").read_text().splitlines()] for s in ("dev", "heldout")}
    return passages, qs


# ----------------------------------------------------------------------------- models (ONNX)
class Onnx:
    def __init__(self, repo, rev):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        self.sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx", revision=rev), so,
                                         providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def run(self, encs):
        feed = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        return self.sess.run(None, feed)[0]


class Dense:
    def __init__(self):
        self.m = Onnx(*MODELS["embed"])

    def encode(self, texts):
        out = self.m.run(self.m.tok.encode_batch(texts))[:, 0]  # CLS pooling (bge)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class CrossEncoder:
    def __init__(self):
        self.m = Onnx(*MODELS["rerank"])

    def score(self, query, texts):
        return self.m.run(self.m.tok.encode_batch([(query, t) for t in texts]))[:, 0]


# ----------------------------------------------------------------------------- BM25 (Okapi)
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = set(tokenize(query))
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf[w] * d[w] * (self.k1 + 1) / (d[w] + norm) for w in q if w in d)
        return s


def rrf(rankings, k):
    fused = Counter()
    for ranking in rankings:
        for r, i in enumerate(ranking):
            fused[i] += 1.0 / (k + r + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]


# ----------------------------------------------------------------------------- systems
class Systems:
    def __init__(self, passages):
        self.passages = passages
        self.texts = [p["text"] for p in passages]
        self.dense = Dense()
        self.ce = CrossEncoder()
        self.pvec = self.dense.encode(self.texts)
        self._bm25 = {}

    def bm25(self, k1, b):
        if (k1, b) not in self._bm25:
            self._bm25[(k1, b)] = BM25(self.texts, k1, b)
        return self._bm25[(k1, b)]

    def rank(self, cfg, query):
        kind = cfg["kind"]
        if kind == "dense":
            q = self.dense.encode([(BGE_QUERY_PREFIX if cfg["prefix"] else "") + query])[0]
            return list(np.argsort(-(self.pvec @ q), kind="stable"))
        if kind == "bm25":
            s = self.bm25(cfg["k1"], cfg["b"]).scores(query)
            return list(np.argsort(-s, kind="stable"))
        if kind == "hybrid":
            d = self.rank(cfg["dense"], query)
            s = self.bm25(cfg["bm25"]["k1"], cfg["bm25"]["b"]).scores(query)
            lexical = [i for i in np.argsort(-s, kind="stable") if s[i] > 0]  # only lexical matches vote
            return rrf([d, lexical], cfg["rrf_k"])
        if kind == "rerank":
            first = self.rank(cfg["first"], query)
            head = first[:cfg["depth"]]
            scores = self.ce.score(query, [self.texts[i] for i in head])
            return [head[j] for j in np.argsort(-scores, kind="stable")] + first[cfg["depth"]:]
        raise ValueError(kind)


def name(cfg):
    k = cfg["kind"]
    if k == "dense":
        return f"dense(bge-small{',query-prefix' if cfg['prefix'] else ''})"
    if k == "bm25":
        return f"bm25(k1={cfg['k1']},b={cfg['b']})"
    if k == "hybrid":
        return f"hybrid-rrf(k={cfg['rrf_k']}; {name(cfg['dense'])} + {name(cfg['bm25'])})"
    return f"rerank(ms-marco-MiniLM-L6, top-{cfg['depth']} of {name(cfg['first'])})"


def evaluate(systems, cfg, questions):
    ids = [p["passage_id"] for p in systems.passages]
    rows, lat = [], []
    for _ in range(2):  # warm-up (untimed)
        systems.rank(cfg, questions[0]["question"])
    for q in questions:
        t = time.perf_counter()
        ranking = systems.rank(cfg, q["question"])
        lat.append((time.perf_counter() - t) * 1000)
        top = [ids[i] for i in ranking[:10]]
        rel = set(q["relevant_passage_ids"])
        hits = len(rel & set(top[:K]))
        first = next((r + 1 for r, p in enumerate(top) if p in rel), None)
        rows.append({"qid": q["qid"], "query_types": q["query_types"], "top10": top,
                     "relevant": sorted(rel), "precision@4": hits / K, "recall@4": hits / len(rel),
                     "rr@10": 1 / first if first else 0.0, "first_relevant_rank": first,
                     "hard_negative_in_top4": sorted(set(q["hard_negative_passage_ids"]) & set(top[:K])),
                     "latency_ms": round(lat[-1], 3)})
    xs = sorted(lat)
    summ = {m: round(statistics.fmean(r[m] for r in rows), 4) for m in ("precision@4", "recall@4", "rr@10")}
    summ.update(latency_p50_ms=round(xs[math.ceil(0.5 * len(xs)) - 1], 3),
                latency_p95_ms=round(xs[math.ceil(0.95 * len(xs)) - 1], 3), n=len(rows))
    return {"system": name(cfg), "config": cfg, "summary": summ, "per_query": rows}


def dev_grid():
    dense = [{"kind": "dense", "prefix": p} for p in (False, True)]
    bm25 = [{"kind": "bm25", "k1": k1, "b": b} for k1 in (0.9, 1.2, 1.5) for b in (0.4, 0.75)]
    return dense, bm25


def select(results):
    return max(results, key=lambda r: (r["summary"]["rr@10"], r["summary"]["recall@4"], -r["summary"]["latency_p50_ms"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], required=True)
    args = ap.parse_args()
    check_labels()
    passages, qs = load()
    systems = Systems(passages)
    RESULTS.mkdir(exist_ok=True)
    env = {"onnxruntime_threads": THREADS, "models": MODELS, "cpu_count": os.cpu_count(),
           "python": sys.version.split()[0], "labels_lock_sha256": hashlib.sha256((EVAL / "LABELS.lock.json").read_bytes()).hexdigest()}
    if args.split == "dev":
        dense_cfgs, bm25_cfgs = dev_grid()
        res = [evaluate(systems, c, qs["dev"]) for c in dense_cfgs + bm25_cfgs]
        best_dense = select([r for r in res if r["config"]["kind"] == "dense"])["config"]
        best_bm25 = select([r for r in res if r["config"]["kind"] == "bm25"])["config"]
        hybrids = [{"kind": "hybrid", "rrf_k": k, "dense": best_dense, "bm25": best_bm25} for k in (10, 30, 60)]
        res += [evaluate(systems, c, qs["dev"]) for c in hybrids]
        best_hybrid = select([r for r in res if r["config"]["kind"] == "hybrid"])["config"]
        reranks = [{"kind": "rerank", "first": f, "depth": d} for f in (best_dense, best_hybrid) for d in (10, 20)]
        res += [evaluate(systems, c, qs["dev"]) for c in reranks]
        best_rerank = select([r for r in res if r["config"]["kind"] == "rerank"])["config"]
        chosen = select(res)
        selection = {"rule": "max dev MRR@10, then dev Recall@4, then lower dev p50 latency",
                     "selected": chosen["config"], "selected_name": chosen["system"],
                     "family_best": {"dense": best_dense, "bm25": best_bm25, "hybrid": best_hybrid, "rerank": best_rerank}}
        (RESULTS / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
        out = {"split": "dev", "environment": env, "results": res, "selection": selection}
    else:
        sel_path = RESULTS / "selection.json"
        if not sel_path.exists():
            sys.exit("run --split dev first: held-out may only evaluate configurations fixed on dev")
        selection = json.loads(sel_path.read_text())
        res = [evaluate(systems, c, qs["heldout"]) for c in selection["family_best"].values()]
        out = {"split": "heldout", "environment": env, "selection": selection, "results": res}
    (RESULTS / f"{args.split}.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in res:
        s = r["summary"]
        print(f"{r['system'][:95]:95s} P@4 {s['precision@4']:.3f} R@4 {s['recall@4']:.3f} MRR {s['rr@10']:.3f} "
              f"p50 {s['latency_p50_ms']:.2f}ms p95 {s['latency_p95_ms']:.2f}ms")
    if args.split == "dev":
        print("SELECTED:", selection["selected_name"])


if __name__ == "__main__":
    main()
