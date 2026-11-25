#!/usr/bin/env python3
"""Offline analysis of the completed live pilot (no API calls): where does the F1 difference come from?"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer_eval as ae  # noqa: E402
import live_eval as le  # noqa: E402
import run_qasper_eval as rq  # noqa: E402

items = {i["question_id"]: i for i in le.pilot_items()}
gen = {(r["config"], r["question_id"]): r for r in le.read_jsonl(le.GEN_LOG) if r.get("answer") is not None}
retr = json.loads((HERE / "results_test.json").read_text())["per_query"]
rec = {cfg: {r["question_id"]: r["recall@4"] for r in retr[f"{k}|T=256|dense"]}
       for cfg, k in (("fixed_T256", "fixed"), ("structure_T256", "structure"))}
rows = []
for qid, it in items.items():
    f, s = gen[("fixed_T256", qid)], gen[("structure_T256", qid)]
    ab = lambda r: ae.normalize_answer(r["answer"]) == "unanswerable"
    rows.append({"qid": qid, "paper_id": it["paper_id"], "type": it["majority_type"],
                 "f1_fixed": ae.answer_f1(f["answer"], it["annotations"]), "f1_struct": ae.answer_f1(s["answer"], it["annotations"]),
                 "abst_fixed": ab(f), "abst_struct": ab(s),
                 "ev_recall_fixed": rec["fixed_T256"].get(qid), "ev_recall_struct": rec["structure_T256"].get(qid)})
out = {"n": len(rows), "by_type": {}, "by_abstention_pattern": {}}
for t in sorted({r["type"] for r in rows}):
    sub = [r for r in rows if r["type"] == t]
    out["by_type"][t] = {"n": len(sub), "f1_fixed": round(statistics.fmean(r["f1_fixed"] for r in sub), 3),
                         "f1_struct": round(statistics.fmean(r["f1_struct"] for r in sub), 3)}
for name, cond in {"both_answered": lambda r: not r["abst_fixed"] and not r["abst_struct"],
                   "only_structure_abstained": lambda r: r["abst_struct"] and not r["abst_fixed"],
                   "only_fixed_abstained": lambda r: r["abst_fixed"] and not r["abst_struct"],
                   "both_abstained": lambda r: r["abst_fixed"] and r["abst_struct"]}.items():
    sub = [r for r in rows if cond(r)]
    out["by_abstention_pattern"][name] = {"n": len(sub), "sum_f1_diff": round(sum(r["f1_struct"] - r["f1_fixed"] for r in sub), 3)}
both = [r for r in rows if not r["abst_fixed"] and not r["abst_struct"]]
out["paired_f1_both_answered"] = rq.paired_bootstrap(
    [{"question_id": r["qid"], "paper_id": r["paper_id"], "v": r["f1_fixed"]} for r in both],
    [{"question_id": r["qid"], "paper_id": r["paper_id"], "v": r["f1_struct"]} for r in both], "v", 10000, 7) if both else None
withrec = [r for r in rows if r["ev_recall_fixed"] is not None]
out["evidence_recall_at4_on_pilot_questions"] = {
    "n": len(withrec), "fixed": round(statistics.fmean(r["ev_recall_fixed"] for r in withrec), 3),
    "structure": round(statistics.fmean(r["ev_recall_struct"] for r in withrec), 3)}
out["note"] = "Post-hoc, descriptive only (n=40); not a new pre-registered test; no live calls."
(HERE / "pilot_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
