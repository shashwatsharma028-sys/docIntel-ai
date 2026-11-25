#!/usr/bin/env python3
"""Live paired QASPER pilot: local BGE retrieval (fixed-256 vs structure-256, top-4) + live Groq
generation, official QASPER answer F1, abstention counts, RAGAS Faithfulness (judge) and optional
LangSmith tracing. This is NOT an evaluation of the full Gemini-FAISS application.

Modes:  --dry-run   retrieval + prompts + token estimates, no network
        --preflight list account models (zero tokens) and check both model IDs are permitted
        --generate  80 generation calls (40 frozen pilot questions x 2 configs), resumable
        --faithfulness  RAGAS Faithfulness on the frozen 19-question subset x 2 configs, resumable
        --report    metrics, paired differences, paper-clustered bootstrap CIs

Safety: hard caps MAX_REQUESTS / MAX_TOTAL_TOKENS (provider-reported usage), <= 3 retries per
call with backoff honouring retry-after, append-only JSONL (completed calls are never repeated),
finish_reason=length triggers one escalation to a larger budget and is recorded. The API key is
read from GROQ_API_KEY (here: a placeholder; the session proxy injects the real secret).
LangSmith traces (LANGSMITH_TRACING=true) contain only public QASPER text and model outputs.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer_eval as ae  # noqa: E402
import run_qasper_eval as rq  # noqa: E402

BASE_URL = "https://api.groq.com/openai/v1"
GEN_MODEL = os.environ.get("DOCINTEL_LIVE_GEN_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.environ.get("DOCINTEL_LIVE_JUDGE_MODEL", "openai/gpt-oss-20b")
GEN = {"temperature": 0, "reasoning_effort": "low", "max_completion_tokens": 1024, "escalated_max": 2048}
MAX_REQUESTS = int(os.environ.get("DOCINTEL_LIVE_MAX_REQUESTS", "200"))
MAX_TOTAL_TOKENS = int(os.environ.get("DOCINTEL_LIVE_MAX_TOKENS", "450000"))
GEN_LOG, FAITH_LOG, USAGE_LOG = HERE / "live_generation.jsonl", HERE / "live_faithfulness.jsonl", HERE / "live_usage.jsonl"
CONFIGS = {"fixed_T256": "fixed", "structure_T256": "structure"}
SYSTEM = ("You answer questions about a scientific paper using only the provided excerpts. "
          "If the excerpts do not contain the answer, reply exactly: Unanswerable. "
          "For yes/no questions reply Yes or No. Otherwise reply with a short answer "
          "(a phrase or one sentence) and nothing else.")

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(**_):
        return lambda f: f


# ----------------------------------------------------------------------------- budget + logs
def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def spent():
    rows = read_jsonl(USAGE_LOG)
    return len(rows), sum(r.get("total_tokens", 0) for r in rows)


def guard():
    n, tok = spent()
    if n >= MAX_REQUESTS or tok >= MAX_TOTAL_TOKENS:
        sys.exit(f"budget reached: {n} requests, {tok} tokens (caps {MAX_REQUESTS}, {MAX_TOTAL_TOKENS})")


def client():
    import httpx
    from openai import OpenAI

    def record(response: "httpx.Response"):
        if response.request.method == "POST" and response.request.url.path.endswith("/chat/completions"):
            response.read()
            try:
                body = response.json()
            except ValueError:
                body = {}
            u = body.get("usage") or {}
            append(USAGE_LOG, {"t": time.time(), "status": response.status_code, "model": body.get("model"),
                               "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                               "total_tokens": u.get("total_tokens", 0),
                               "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                               "ratelimit": {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit")}})
    http = httpx.Client(timeout=60, event_hooks={"response": [record]})
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get("GROQ_API_KEY", "proxy-injected"), http_client=http, max_retries=0)


def with_retries(fn, tries=3):
    import openai
    for attempt in range(tries + 1):
        guard()
        try:
            return fn()
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            if attempt == tries:
                raise
            wait = float(getattr(getattr(e, "response", None), "headers", {}).get("retry-after", 2 ** (attempt + 1)))
            time.sleep(min(wait, 60))


# ----------------------------------------------------------------------------- retrieval
def pilot_items():
    aset = json.loads((HERE / "answer_set.json").read_text())
    by = {q["question_id"]: q for q in aset["questions"]}
    rows = {r["id"]: r for r in rq.load_split("test")}
    items = []
    for qid in aset["pilot"]["question_ids"]:
        q = by[qid]
        qa = rows[q["paper_id"]]["qas"]
        i = qa["question_id"].index(qid)
        items.append({**q, "annotations": qa["answers"][i]["answer"], "row": rows[q["paper_id"]]})
    return items


class Retriever:
    def __init__(self):
        self.dense = rq.rre.Dense()
        from tokenizers import Tokenizer
        raw = Tokenizer.from_str(self.dense.m.tok.to_str())
        raw.no_truncation()
        raw.no_padding()
        self.tok = type("T", (), {"n": staticmethod(lambda t: len(raw.encode(t, add_special_tokens=False).ids))})
        self._cache = {}

    @traceable(name="retrieve", run_type="retriever")
    def top4(self, paper_row, kind, question):
        key = (paper_row["id"], kind)
        if key not in self._cache:
            ch = rq.Paper(paper_row, self.tok).chunks(kind, 256)
            self._cache[key] = (ch, self.dense.encode(ch))
        ch, vec = self._cache[key]
        order = np.argsort(-(vec @ self.dense.encode([question])[0]), kind="stable")[:4]
        return [ch[i] for i in order]


def messages(question, contexts):
    ctx = "\n\n".join(f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(contexts))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"}]


# ----------------------------------------------------------------------------- stages
@traceable(name="generate", run_type="llm")
def generate(cl, msgs):
    for limit in (GEN["max_completion_tokens"], GEN["escalated_max"]):
        r = with_retries(lambda: cl.chat.completions.create(model=GEN_MODEL, messages=msgs, temperature=GEN["temperature"],
                                                            reasoning_effort=GEN["reasoning_effort"], max_completion_tokens=limit))
        choice = r.choices[0]
        if choice.finish_reason != "length":
            break
    return {"answer": (choice.message.content or "").strip(), "finish_reason": choice.finish_reason, "limit_used": limit,
            "model": r.model, "usage": r.usage.model_dump() if r.usage else None}


def run_generation(dry=False):
    done = {(r["config"], r["question_id"]) for r in read_jsonl(GEN_LOG) if r.get("answer") is not None}
    ret, cl = Retriever(), (None if dry else client())
    est = 0
    for item in pilot_items():
        for cfg, kind in CONFIGS.items():
            if (cfg, item["question_id"]) in done:
                continue
            t0 = time.perf_counter()
            ctx = ret.top4(item["row"], kind, item["question"])
            t_ret = (time.perf_counter() - t0) * 1000
            msgs = messages(item["question"], ctx)
            if dry:
                est += int(1.15 * sum(ret.tok.n(m["content"]) for m in msgs))
                continue
            t1 = time.perf_counter()
            try:
                out = generate(cl, msgs)
                err = None
            except Exception as e:  # recorded, not retried beyond the bounded policy
                out, err = {"answer": None}, f"{type(e).__name__}: {e}"[:300]
            t_gen = (time.perf_counter() - t1) * 1000
            append(GEN_LOG, {"config": cfg, "question_id": item["question_id"], "paper_id": item["paper_id"],
                             "contexts": ctx, "retrieval_ms": t_ret, "generation_ms": t_gen,
                             "end_to_end_ms": t_ret + t_gen, "error": err, **out})
            print(cfg, item["question_id"], (out.get("answer") or err or "")[:60], flush=True)
    if dry:
        print(f"dry run: {2 * len(pilot_items())} prompts, ~{est} input tokens (BGE x1.15 estimate)")


def run_faithfulness():
    import asyncio
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from openai import AsyncOpenAI
    import httpx
    subset = set(json.loads((HERE / "faithfulness_subset.json").read_text())["question_ids"])
    gens = {(r["config"], r["question_id"]): r for r in read_jsonl(GEN_LOG) if r.get("answer")}
    done = {(r["config"], r["question_id"]) for r in read_jsonl(FAITH_LOG) if r.get("score") is not None}
    sync_hook = client()._client.event_hooks["response"][0]

    async def ahook(resp):
        await resp.aread()
        sync_hook(resp)
    acl = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("GROQ_API_KEY", "proxy-injected"), max_retries=2,
                      http_client=httpx.AsyncClient(timeout=90, event_hooks={"response": [ahook]}))
    metric = Faithfulness(llm=llm_factory(JUDGE_MODEL, provider="openai", client=acl, max_tokens=2048))
    for (cfg, qid), g in sorted(gens.items()):
        if qid not in subset or (cfg, qid) in done:
            continue
        guard()
        try:
            res = asyncio.run(metric.ascore(user_input=next(i["question"] for i in pilot_items() if i["question_id"] == qid),
                                            response=g["answer"], retrieved_contexts=g["contexts"]))
            row = {"config": cfg, "question_id": qid, "paper_id": g["paper_id"], "score": float(res.value), "error": None}
        except Exception as e:
            row = {"config": cfg, "question_id": qid, "paper_id": g["paper_id"], "score": None, "error": f"{type(e).__name__}: {e}"[:300]}
        append(FAITH_LOG, row)
        print(cfg, qid, row["score"], row["error"], flush=True)


def report():
    items = {i["question_id"]: i for i in pilot_items()}
    gens = [r for r in read_jsonl(GEN_LOG) if r.get("answer") is not None]
    by = {(r["config"], r["question_id"]): r for r in gens}
    out = {"generation_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "generation_settings": GEN, "configs": {}}
    per_q = {}
    for cfg in CONFIGS:
        rows = [r for r in gens if r["config"] == cfg]
        for r in rows:
            anns = items[r["question_id"]]["annotations"]
            r["f1"] = ae.answer_f1(r["answer"], anns)
            gold_unans = all(a["unanswerable"] for a in anns)
            r["pred_unans"] = ae.normalize_answer(r["answer"]) == "unanswerable"
            r["gold_unans"] = gold_unans
            per_q[(cfg, r["question_id"])] = r
        lat = sorted(r["end_to_end_ms"] for r in rows)
        pct = lambda q: round(lat[math.ceil(q * len(lat)) - 1], 1) if lat else None
        u = [r["usage"] for r in rows if r.get("usage")]
        out["configs"][cfg] = {
            "n": len(rows), "answer_f1": round(statistics.fmean(r["f1"] for r in rows), 4) if rows else None,
            "abstained": sum(r["pred_unans"] for r in rows), "gold_unanswerable": sum(r["gold_unans"] for r in rows),
            "correct_abstentions": sum(r["pred_unans"] and r["gold_unans"] for r in rows),
            "truncated_finish_length": sum(r["finish_reason"] == "length" for r in rows),
            "prompt_tokens_mean": round(statistics.fmean(x["prompt_tokens"] for x in u), 1) if u else None,
            "completion_tokens_mean": round(statistics.fmean(x["completion_tokens"] for x in u), 1) if u else None,
            "end_to_end_p50_ms": pct(0.5), "end_to_end_p95_ms": pct(0.95)}
    faith = {(r["config"], r["question_id"]): r for r in read_jsonl(FAITH_LOG) if r.get("score") is not None}
    for cfg in CONFIGS:
        s = [r["score"] for (c, _), r in faith.items() if c == cfg]
        out["configs"][cfg]["faithfulness_mean"] = round(statistics.fmean(s), 4) if s else None
        out["configs"][cfg]["faithfulness_n"] = len(s)

    def paired(metric, table):
        rows_a, rows_b = [], []
        for (cfg, qid), r in table.items():
            if cfg == "fixed_T256" and ("structure_T256", qid) in table:
                rows_a.append({"question_id": qid, "paper_id": r["paper_id"], "v": r[metric]})
                rows_b.append({"question_id": qid, "paper_id": r["paper_id"], "v": table[("structure_T256", qid)][metric]})
        return rq.paired_bootstrap(rows_a, rows_b, "v", 10000, 7) if rows_a else None
    out["paired_structure_minus_fixed"] = {"answer_f1": paired("f1", per_q), "faithfulness": paired("score", faith),
                                           "prompt_tokens": paired("pt", {k: {**v, "pt": v["usage"]["prompt_tokens"]}
                                                                          for k, v in per_q.items() if v.get("usage")})}
    n, tok = spent()
    out["provider_usage_total"] = {"requests": n, "total_tokens": tok}
    (HERE / "live_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def preflight():
    cl = client()
    ids = sorted(m.id for m in cl.models.list().data)
    ok = {m: m in ids for m in (GEN_MODEL, JUDGE_MODEL)}
    print(json.dumps({"permitted": ok, "n_models": len(ids), "gpt_oss_like": [i for i in ids if "oss" in i]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("dry-run", "preflight", "generate", "faithfulness", "report"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        run_generation(dry=True)
    if a.preflight:
        preflight()
    if a.generate:
        run_generation()
    if a.faithfulness:
        run_faithfulness()
    if a.report:
        report()
