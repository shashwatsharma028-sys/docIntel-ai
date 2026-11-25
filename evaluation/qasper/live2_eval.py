#!/usr/bin/env python3
"""Stage 2 live comparison (frozen in live2_criteria.json): A = structure-aware T=256 dense top-4 vs
REF = dense top-20 reranked by the MiniLM cross-encoder, top-4. Same generator, prompt and settings
as the earlier pilot (live_eval.py). 123 frozen QASPER test questions. Live Groq generation over LOCAL
BGE retrieval -- not an evaluation of the full Gemini-FAISS application.

  --plan      verify reuse compatibility, count planned requests (no network)
  --generate  resumable; budget/quota guarded (live_eval ledger machinery, separate files)
  --report    official answer F1, paired paper-clustered CIs, tokens, abstention, latency
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer_eval as ae  # noqa: E402
import live_eval as le  # noqa: E402
import run_qasper_eval as rq  # noqa: E402

le.LEDGER = HERE / "live2_attempts.jsonl"
GEN2 = HERE / "live2_generation.jsonl"
CRIT = json.loads((HERE / "live2_criteria.json").read_text())
le.MAX_REQUESTS = 260          # 211 planned + bounded escalations/retries
le.MAX_TOTAL_TOKENS = 300000
SYSTEM_SHA = hashlib.sha256(le.SYSTEM.encode()).hexdigest()
_pacing = threading.local()
_real_sleep = time.sleep


def _counting_sleep(s):
    _pacing.ms = getattr(_pacing, "ms", 0.0) + s * 1000
    _real_sleep(s)


le.time.sleep = _counting_sleep  # rate-limit pacing time is recorded and reported separately


def questions():
    frozen = json.loads(rq.FROZEN.read_text())
    rows = {r["id"]: r for r in rq.load_split("test")}
    out = []
    for p in frozen["test"]["papers"]:
        qa = rows[p["paper_id"]]["qas"]
        for q in p["questions"]:
            i = qa["question_id"].index(q["question_id"])
            out.append({"question_id": q["question_id"], "paper_id": p["paper_id"], "question": q["question"],
                        "annotations": qa["answers"][i]["answer"], "row": rows[p["paper_id"]]})
    return out


class Retrieval:
    def __init__(self):
        self.r = le.Retriever()
        self.ce = rq.rre.CrossEncoder()
        self._c = {}

    def contexts(self, item, cfg):
        key = item["paper_id"]
        if key not in self._c:
            ch = rq.Paper(item["row"], self.r.tok).chunks("structure", 256)
            self._c[key] = (ch, self.r.dense.encode(ch))
        ch, vec = self._c[key]
        t = time.perf_counter()
        d = list(np.argsort(-(vec @ self.r.dense.encode([item["question"]])[0]), kind="stable"))
        if cfg == "A":
            out = [ch[i] for i in d[:4]]
        else:
            head = d[:20]
            sc = self.ce.score(item["question"], [ch[i] for i in head])
            out = [ch[head[j]] for j in np.argsort(-sc, kind="stable")[:4]]
        return out, (time.perf_counter() - t) * 1000


def reusable(items, ret):
    """Pilot structure_T256 answers are reused only if contexts, prompt and settings match exactly."""
    pilot = {r["question_id"]: r for r in le.read_jsonl(le.GEN_LOG)
             if r["config"] == "structure_T256" and r.get("answer") is not None}
    ok, rejected = {}, {}
    for it in items:
        r = pilot.get(it["question_id"])
        if r is None:
            continue
        ctx, _ = ret.contexts(it, "A")
        checks = {"contexts_identical": ctx == r["contexts"], "model": r.get("model"),
                  "limit_used": r.get("limit_used"), "finish_reason": r.get("finish_reason")}
        if checks["contexts_identical"] and r.get("model") == le.GEN_MODEL and r.get("limit_used") == le.GEN["max_completion_tokens"]:
            ok[it["question_id"]] = {**r, "reused_from_pilot": True, "reuse_checks": checks}
        else:
            rejected[it["question_id"]] = checks
    return ok, rejected


def plan():
    items, ret = questions(), Retrieval()
    ok, rejected = reusable(items, ret)
    print(json.dumps({"questions": len(items), "A_reusable": len(ok), "A_rejected": rejected,
                      "A_new": len(items) - len(ok), "REF_new": len(items),
                      "planned_requests": 2 * len(items) - len(ok), "system_prompt_sha256": SYSTEM_SHA}, indent=1))


def generate():
    items, ret = questions(), Retrieval()
    ok, _ = reusable(items, ret)
    done = {(r["config"], r["question_id"]) for r in le.read_jsonl(GEN2) if r.get("answer") is not None}
    for qid, r in ok.items():
        if ("A", qid) not in done:
            le.append(GEN2, {"config": "A", "question_id": qid, "paper_id": r["paper_id"], "answer": r["answer"],
                             "contexts": r["contexts"], "model": r["model"], "usage": r["usage"],
                             "finish_reason": r["finish_reason"], "limit_used": r["limit_used"],
                             "end_to_end_ms": r["end_to_end_ms"], "retrieval_ms": r.get("retrieval_ms"),
                             "pacing_ms": None, "reused_from_pilot": True, "reuse_checks": r["reuse_checks"]})
            done.add(("A", qid))
    le.CLIENT = le.client()
    le.PLANNED.update(generation=2 * len(items) - len(ok), faithfulness_est=0)
    sent = 0
    for n, it in enumerate(items):
        for cfg in (("A", "REF") if n % 2 == 0 else ("REF", "A")):  # alternate order against time drift
            if (cfg, it["question_id"]) in done:
                continue
            ctx, t_ret = ret.contexts(it, cfg)
            _pacing.ms = 0.0
            t = time.perf_counter()
            try:
                out, err = le.traced_generate(it["question"], ctx, le.GEN_MODEL, le.GEN), None
            except Exception as e:
                if le.budget_cause(e):
                    raise le.budget_cause(e)
                out, err = {"answer": None}, f"{type(e).__name__}: {e}"[:300]
            t_gen = (time.perf_counter() - t) * 1000
            sent += 1
            le.append(GEN2, {"config": cfg, "question_id": it["question_id"], "paper_id": it["paper_id"], "error": err,
                             "contexts": ctx, "retrieval_ms": t_ret, "generation_ms": t_gen, "pacing_ms": _pacing.ms,
                             "end_to_end_ms": t_ret + t_gen, "reused_from_pilot": False, **out})
            print(cfg, it["question_id"][:10], repr((out.get("answer") or err or "")[:50]), out.get("finish_reason"), flush=True)
            if not le.STATE["quota_checked"] and out.get("answer") is not None:
                le.quota_check(0)


def report():
    items = {i["question_id"]: i for i in questions()}
    latest = {}
    for r in le.read_jsonl(GEN2):
        latest[(r["config"], r["question_id"])] = r
    scored, cfgs = {}, {}
    for cfg in ("A", "REF"):
        rows = [latest.get((cfg, q)) for q in items]
        ok = [r for r in rows if r and r.get("answer") is not None]
        for r in ok:
            anns = items[r["question_id"]]["annotations"]
            scored[(cfg, r["question_id"])] = {**r, "f1": ae.answer_f1(r["answer"], anns),
                                               "abst": ae.normalize_answer(r["answer"]) == "unanswerable"}
        sr = [scored[(cfg, r["question_id"])] for r in ok]
        new = [r for r in sr if not r.get("reused_from_pilot")]
        e2e = sorted(r["end_to_end_ms"] for r in new)
        nopace = sorted(r["end_to_end_ms"] - (r.get("pacing_ms") or 0) for r in new)
        pct = lambda xs, q: round(xs[math.ceil(q * len(xs)) - 1], 1) if xs else None
        u = [r["usage"] for r in new if r.get("usage")]
        cfgs[cfg] = {"planned": len(items), "answered": len(ok), "failed": sum(1 for r in rows if r and r.get("answer") is None),
                     "missing": sum(1 for r in rows if r is None), "reused_from_pilot": sum(1 for r in sr if r.get("reused_from_pilot")),
                     "truncated": sum(r.get("finish_reason") == "length" for r in sr),
                     "answer_f1": round(statistics.fmean(r["f1"] for r in sr), 4) if sr else None,
                     "abstained": sum(r["abst"] for r in sr),
                     "new_responses_prompt_tokens_mean": round(statistics.fmean(x["prompt_tokens"] for x in u), 1) if u else None,
                     "new_responses_completion_tokens_mean": round(statistics.fmean(x["completion_tokens"] for x in u), 1) if u else None,
                     "retrieval_ms_p50": pct(sorted(r["retrieval_ms"] for r in new), 0.5),
                     "end_to_end_ms_p50": pct(e2e, 0.5), "end_to_end_ms_p95": pct(e2e, 0.95),
                     "end_to_end_excl_pacing_ms_p50": pct(nopace, 0.5), "end_to_end_excl_pacing_ms_p95": pct(nopace, 0.95)}

    def paired(only_new=False):
        a, b = [], []
        for qid in items:
            ra, rb = scored.get(("A", qid)), scored.get(("REF", qid))
            if ra and rb and not (only_new and ra.get("reused_from_pilot")):
                a.append({"question_id": qid, "paper_id": ra["paper_id"], "v": ra["f1"]})
                b.append({"question_id": qid, "paper_id": rb["paper_id"], "v": rb["f1"]})
        res = rq.paired_bootstrap(a, b, "v", 10000, 7) if a else None
        if res:
            lo, hi = res["ci95"]
            res["verdict"] = "improved" if lo > 0 else "non-inferior" if lo > -0.05 else "degraded" if hi < -0.05 else "inconclusive"
        return res
    led = le.read_jsonl(le.LEDGER)
    resp = [r for r in led if r["event"] == "response"]
    out = {"scope": "live Groq generation over local BGE retrieval on QASPER text; NOT the full Gemini-FAISS application",
           "criteria": CRIT, "system_prompt_sha256": SYSTEM_SHA, "configs": cfgs,
           "primary_REF_minus_A_answer_f1": paired(),
           "sensitivity_new_responses_only": paired(only_new=True),
           "provider_usage_stage2": {"attempted_requests": sum(r["event"] == "attempt" for r in led), "responses": len(resp),
                                     "http_errors": sum(r["status"] >= 400 for r in resp),
                                     "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in resp),
                                     "completion_tokens": sum(r.get("completion_tokens", 0) for r in resp),
                                     "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in resp),
                                     "total_tokens": sum(r.get("total_tokens", 0) for r in resp),
                                     "models_reported": sorted({r.get("model") for r in resp if r.get("model")}),
                                     "quota_checks": [r for r in led if r["event"] == "quota_check"]}}
    (HERE / "live2_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "criteria"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "generate", "report"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    try:
        if a.plan:
            plan()
        if a.generate:
            generate()
        if a.report:
            report()
    except le.BudgetStop as e:
        print("STOPPED:", e)
        sys.exit(2)
