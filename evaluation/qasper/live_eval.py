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
import re
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
GEN_LOG, FAITH_LOG = HERE / "live_generation.jsonl", HERE / "live_faithfulness.jsonl"
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


# ----------------------------------------------------------------------------- ledger, budgets, quota
LEDGER = HERE / "live_attempts.jsonl"
STATE = {"headers": {}, "quota_checked": False}
PROJECT = os.environ.get("LANGSMITH_PROJECT", "docintel-qasper-pilot")
PLANNED = {"generation": 80, "faithfulness_est": 76}


class BudgetStop(RuntimeError):
    pass


def budget_cause(exc):
    """The OpenAI SDK wraps exceptions raised in HTTP hooks as APIConnectionError; recover BudgetStop."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, BudgetStop):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def ledger_stats():
    rows = read_jsonl(LEDGER)
    return (sum(r["event"] == "attempt" for r in rows),
            sum(r.get("total_tokens", 0) for r in rows if r["event"] == "response"))


def _reset_seconds(v):
    total = 0.0
    for num, unit in re.findall(r"([0-9.]+)(ms|s|m|h)", v or ""):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def _before_request(request):
    """Runs before EVERY HTTP request (generation, judge, internal retries): budget + pacing + count."""
    if not request.url.path.endswith("/chat/completions"):
        return
    body = json.loads(request.content or b"{}")
    prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    reserve = prompt_chars // 3 + int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)
    n, used = ledger_stats()
    if n + 1 > MAX_REQUESTS or used + reserve > MAX_TOTAL_TOKENS:
        raise BudgetStop(f"budget: {n} attempts, {used} tokens used, reservation {reserve} "
                         f"(caps {MAX_REQUESTS} requests, {MAX_TOTAL_TOKENS} tokens)")
    h = STATE["headers"]
    if h.get("x-ratelimit-remaining-tokens") and int(h["x-ratelimit-remaining-tokens"]) < reserve:
        time.sleep(_reset_seconds(h.get("x-ratelimit-reset-tokens")) + 1)  # per-minute window pacing
    attempt_id = f"{time.time():.6f}-{n + 1}"
    request.extensions["attempt_id"] = attempt_id
    append(LEDGER, {"event": "attempt", "id": attempt_id, "t": time.time(), "model": body.get("model"), "reserved": reserve})


def _after_response(response, body):
    STATE["headers"] = {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit")}
    u = (body or {}).get("usage") or {}
    append(LEDGER, {"event": "response", "id": response.request.extensions.get("attempt_id"), "status": response.status_code,
                    "model": (body or {}).get("model"), "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0), "total_tokens": u.get("total_tokens", 0),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "error": None if response.status_code < 400 else str((body or {}).get("error", ""))[:200],
                    "ratelimit": STATE["headers"]})


def _sync_response(response):
    if response.request.url.path.endswith("/chat/completions"):
        response.read()
        try:
            _after_response(response, response.json())
        except ValueError:
            _after_response(response, None)


async def _async_request(request):
    _before_request(request)


async def _async_response(response):
    if response.request.url.path.endswith("/chat/completions"):
        await response.aread()
        try:
            _after_response(response, response.json())
        except ValueError:
            _after_response(response, None)


def client():
    import httpx
    from openai import OpenAI
    http = httpx.Client(timeout=90, event_hooks={"request": [_before_request], "response": [_sync_response]})
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get("GROQ_API_KEY", "proxy-injected"), http_client=http, max_retries=0)


def async_client():
    import httpx
    from openai import AsyncOpenAI
    http = httpx.AsyncClient(timeout=120, event_hooks={"request": [_async_request], "response": [_async_response]})
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("GROQ_API_KEY", "proxy-injected"), http_client=http, max_retries=0)


def quota_check(done_generations):
    """Frozen rule: continue only if remaining daily requests >= 1.25 x remaining planned requests."""
    h = STATE["headers"]
    remaining = int(h.get("x-ratelimit-remaining-requests", -1))
    planned = (PLANNED["generation"] - done_generations) + PLANNED["faithfulness_est"]
    ok = remaining >= 1.25 * planned
    row = {"event": "quota_check", "t": time.time(), "headers": h, "remaining_requests": remaining,
           "planned_remaining": planned, "continue": ok}
    append(LEDGER, row)
    print("QUOTA CHECK", json.dumps(row), flush=True)
    STATE["quota_checked"] = True
    if not ok:
        raise BudgetStop("quota rule failed; stopping before further calls")


def with_retries(fn, tries=3):
    import openai
    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception as e:  # never retry or mask a budget stop
            stop = budget_cause(e)
            if stop is not None:
                raise stop
            if not isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                                  openai.InternalServerError)):
                raise
            err = e
        try:
            raise err
        except openai.RateLimitError as e:
            msg = str(e).lower()
            if "per day" in msg or "tpd" in msg or "rpd" in msg or "upgrade" in msg:
                raise BudgetStop(f"daily/free-tier limit reached: {str(e)[:200]}")
            if attempt == tries:
                raise
            time.sleep(min(_reset_seconds(e.response.headers.get("retry-after", "") + "s") or 2 ** (attempt + 1), 60))
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            if attempt == tries:
                raise
            time.sleep(2 ** (attempt + 1))


# ----------------------------------------------------------------------------- tracing (public data only)
def _sanitize(d):
    return {k: v for k, v in (d or {}).items() if not re.search(r"key|secret|token|auth|client|password", k, re.I)}


try:
    import langsmith
    from langsmith import traceable as _ls_traceable
    _TRACING = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    _LS_CLIENT = langsmith.Client(hide_inputs=_sanitize, hide_outputs=_sanitize) if _TRACING else None

    def traceable(**kw):
        return _ls_traceable(client=_LS_CLIENT, project_name=PROJECT, **kw) if _TRACING else (lambda f: f)
except ImportError:  # pragma: no cover
    def traceable(**_):
        return lambda f: f


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
RET, ROWS, CLIENT = None, {}, None


@traceable(name="retrieve", run_type="retriever")
def traced_retrieve(question: str, paper_id: str, config: str) -> list:
    return RET.top4(ROWS[paper_id], CONFIGS[config], question)


@traceable(name="generate", run_type="llm")
def traced_generate(question: str, contexts: list, model: str, settings: dict) -> dict:
    msgs = messages(question, contexts)
    for limit in (settings["max_completion_tokens"], settings["escalated_max"]):
        r = with_retries(lambda: CLIENT.chat.completions.create(
            model=model, messages=msgs, temperature=settings["temperature"],
            reasoning_effort=settings["reasoning_effort"], max_completion_tokens=limit))
        choice = r.choices[0]
        if choice.finish_reason != "length":
            break
    return {"answer": (choice.message.content or "").strip(), "finish_reason": choice.finish_reason,
            "limit_used": limit, "model": r.model, "usage": r.usage.model_dump() if r.usage else None}


@traceable(name="qasper_pilot_question", run_type="chain")
def traced_pipeline(question: str, paper_id: str, config: str) -> dict:
    t0 = time.perf_counter()
    ctx = traced_retrieve(question, paper_id, config)
    t_ret = (time.perf_counter() - t0) * 1000
    out = traced_generate(question, ctx, GEN_MODEL, GEN)
    return {**out, "contexts": ctx, "retrieval_ms": t_ret, "end_to_end_ms": (time.perf_counter() - t0) * 1000}


def run_generation(dry=False, limit=None):
    global RET, CLIENT
    rows = read_jsonl(GEN_LOG)
    done = {(r["config"], r["question_id"]) for r in rows if r.get("answer") is not None}
    RET = Retriever()
    CLIENT = None if dry else client()
    est, sent = 0, 0
    items = pilot_items()
    for item in items:
        ROWS[item["paper_id"]] = item["row"]
    for item in items:
        for cfg in CONFIGS:
            if (cfg, item["question_id"]) in done:
                continue
            if dry:
                ctx = RET.top4(item["row"], CONFIGS[cfg], item["question"])
                est += int(1.15 * sum(RET.tok.n(m["content"]) for m in messages(item["question"], ctx)))
                continue
            if limit is not None and sent >= limit:
                return
            try:
                out, err = traced_pipeline(item["question"], item["paper_id"], cfg), None
            except Exception as e:
                if budget_cause(e):
                    raise budget_cause(e)
                out, err = {"answer": None}, f"{type(e).__name__}: {e}"[:300]
            sent += 1
            append(GEN_LOG, {"config": cfg, "question_id": item["question_id"], "paper_id": item["paper_id"],
                             "error": err, **out})
            print(cfg, item["question_id"], repr((out.get("answer") or err or "")[:70]), out.get("finish_reason"), flush=True)
            if not STATE["quota_checked"] and out.get("answer") is not None:
                quota_check(len(done) + sent)
    if dry:
        print(f"dry run: {2 * len(items)} prompts, ~{est} input tokens (BGE x1.15 estimate)")


def run_faithfulness(limit=None):
    import asyncio
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    subset = set(json.loads((HERE / "faithfulness_subset.json").read_text())["question_ids"])
    questions = {i["question_id"]: i["question"] for i in pilot_items()}
    gens = {(r["config"], r["question_id"]): r for r in read_jsonl(GEN_LOG) if r.get("answer")}
    done = {(r["config"], r["question_id"]) for r in read_jsonl(FAITH_LOG) if r.get("score") is not None}
    judge = llm_factory(JUDGE_MODEL, provider="openai", client=async_client(), max_tokens=4096, reasoning_effort="low")
    metric = Faithfulness(llm=judge)

    @traceable(name="ragas_faithfulness", run_type="chain")
    def traced_score(question: str, answer: str, contexts: list) -> dict:
        return {"score": float(asyncio.run(metric.ascore(user_input=question, response=answer, retrieved_contexts=contexts)).value)}
    n = 0
    for (cfg, qid), g in sorted(gens.items()):
        if qid not in subset or (cfg, qid) in done:
            continue
        if limit is not None and n >= limit:
            return
        try:
            row = {"score": traced_score(questions[qid], g["answer"], g["contexts"])["score"], "error": None}
        except Exception as e:
            if budget_cause(e):
                raise budget_cause(e)
            row = {"score": None, "error": f"{type(e).__name__}: {e}"[:300]}
        n += 1
        append(FAITH_LOG, {"config": cfg, "question_id": qid, "paper_id": g["paper_id"], **row})
        print("faith", cfg, qid, row["score"], row["error"], flush=True)


def report():
    items = {i["question_id"]: i for i in pilot_items()}
    crit = json.loads((HERE / "live_criteria.json").read_text())
    gen_rows = read_jsonl(GEN_LOG)
    latest = {}
    for r in gen_rows:  # last record per (config, question) wins (a later success supersedes a failure)
        latest[(r["config"], r["question_id"])] = r
    out = {"scope": "live Groq generation over local BGE retrieval on QASPER text; NOT the full Gemini-FAISS application",
           "generation_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "generation_settings": GEN, "configs": {}}
    scored = {}
    for cfg in CONFIGS:
        rows = [latest.get((cfg, q)) for q in items]
        ok = [r for r in rows if r and r.get("answer") is not None]
        for r in ok:
            anns = items[r["question_id"]]["annotations"]
            scored[(cfg, r["question_id"])] = {**r, "f1": ae.answer_f1(r["answer"], anns),
                                               "pred_unans": ae.normalize_answer(r["answer"]) == "unanswerable",
                                               "gold_unans": all(a["unanswerable"] for a in anns)}
        sr = [scored[(cfg, r["question_id"])] for r in ok]
        lat = sorted(r["end_to_end_ms"] for r in ok)
        pct = lambda q: round(lat[math.ceil(q * len(lat)) - 1], 1) if lat else None
        u = [r["usage"] for r in ok if r.get("usage")]
        out["configs"][cfg] = {
            "planned": len(items), "answered": len(ok), "failed": sum(1 for r in rows if r and r.get("answer") is None),
            "missing": sum(1 for r in rows if r is None),
            "truncated_after_escalation": sum(r.get("finish_reason") == "length" for r in ok),
            "escalated_to_2048": sum(r.get("limit_used") == GEN["escalated_max"] for r in ok),
            "answer_f1_mean": round(statistics.fmean(r["f1"] for r in sr), 4) if sr else None,
            "abstained": sum(r["pred_unans"] for r in sr), "gold_unanswerable": sum(r["gold_unans"] for r in sr),
            "correct_abstentions": sum(r["pred_unans"] and r["gold_unans"] for r in sr),
            "prompt_tokens_mean": round(statistics.fmean(x["prompt_tokens"] for x in u), 1) if u else None,
            "completion_tokens_mean": round(statistics.fmean(x["completion_tokens"] for x in u), 1) if u else None,
            "reasoning_tokens_mean": round(statistics.fmean((x.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for x in u), 1) if u else None,
            "end_to_end_p50_ms": pct(0.5), "end_to_end_p95_ms": pct(0.95)}
    faith_rows = read_jsonl(FAITH_LOG)
    flatest = {}
    for r in faith_rows:
        flatest[(r["config"], r["question_id"])] = r
    faith = {k: v for k, v in flatest.items() if v.get("score") is not None}
    for cfg in CONFIGS:
        s_ = [v["score"] for (c, _), v in faith.items() if c == cfg]
        out["configs"][cfg].update(faithfulness_mean=round(statistics.fmean(s_), 4) if s_ else None,
                                   faithfulness_n=len(s_),
                                   faithfulness_failed=sum(1 for (c, _), v in flatest.items() if c == cfg and v.get("score") is None))

    def paired(table, key):
        a, b = [], []
        for (cfg, qid), r in table.items():
            if cfg == "fixed_T256" and ("structure_T256", qid) in table:
                a.append({"question_id": qid, "paper_id": r["paper_id"], "v": r[key]})
                b.append({"question_id": qid, "paper_id": r["paper_id"], "v": table[("structure_T256", qid)][key]})
        return rq.paired_bootstrap(a, b, "v", 10000, 7) if a else None
    f1 = paired(scored, "f1")
    if f1:
        lo, hi = f1["ci95"]
        m = crit["answer_f1"]["non_inferiority_margin"]
        f1["verdict"] = "better" if lo > 0 else "non-inferior" if lo > -m else "degraded" if hi < -m else "inconclusive"
    out["paired_structure_minus_fixed"] = {
        "answer_f1": f1, "faithfulness": paired(faith, "score"),
        "prompt_tokens": paired({k: {**v, "pt": v["usage"]["prompt_tokens"]} for k, v in scored.items() if v.get("usage")}, "pt")}
    led = read_jsonl(LEDGER)
    resp = [r for r in led if r["event"] == "response"]
    by_model = {}
    for r in resp:
        d = by_model.setdefault(r.get("model") or "unknown", {"responses": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                               "total_tokens": 0, "reasoning_tokens": 0, "http_errors": 0})
        d["responses"] += 1
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            d[k] += r.get(k, 0)
        d["reasoning_tokens"] += r.get("reasoning_tokens") or 0
        d["http_errors"] += r["status"] >= 400
    out["provider_usage"] = {"attempted_requests": sum(r["event"] == "attempt" for r in led),
                             "attempts_without_response": sum(r["event"] == "attempt" for r in led) - len(resp),
                             "by_model": by_model, "quota_checks": [r for r in led if r["event"] == "quota_check"]}
    (HERE / "live_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "provider_usage"} | {"usage": out["provider_usage"]["by_model"],
                     "attempted": out["provider_usage"]["attempted_requests"]}, indent=1))


def verify_traces():
    import langsmith
    cl = langsmith.Client()
    roots = list(cl.list_runs(project_name=PROJECT, is_root=True))
    allruns = list(cl.list_runs(project_name=PROJECT))
    names = {}
    for r in allruns:
        names[r.name] = names.get(r.name, 0) + 1
    res = {"project": PROJECT, "root_traces": len(roots), "runs_total": len(allruns), "runs_by_name": names,
           "errored_runs": sum(1 for r in allruns if r.error)}
    (HERE / "live_traces.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def preflight():
    cl = client()
    ids = sorted(m.id for m in cl.models.list().data)
    ok = {m: m in ids for m in (GEN_MODEL, JUDGE_MODEL)}
    print(json.dumps({"permitted": ok, "n_models": len(ids), "gpt_oss_like": [i for i in ids if "oss" in i]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("dry-run", "preflight", "generate", "faithfulness", "report", "verify-traces"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    try:
        if a.dry_run:
            run_generation(dry=True)
        if a.preflight:
            preflight()
        if a.generate:
            run_generation(limit=a.limit)
        if a.faithfulness:
            run_faithfulness(limit=a.limit)
        if a.report:
            report()
        if a.verify_traces:
            verify_traces()
    except BudgetStop as e:
        print("STOPPED:", e)
        sys.exit(2)
