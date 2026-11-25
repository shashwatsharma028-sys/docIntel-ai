#!/usr/bin/env python3
"""Answer-level QASPER evaluation: OFFLINE PREPARATION ONLY (no generation is run here).

  --freeze  answer-level question set from the 38 frozen TEST papers, all answer types including
            unanswerable; seeded 40-question paired pilot -> answer_set.json
  --plan    request counts and token estimates for approval (uses measured retrieved-context sizes)

`answer_f1` ports the official QASPER answer scoring (allenai/qasper-led-baseline
qasper_evaluator.py): SQuAD-style normalisation and token F1, references per annotation
(extractive spans joined with ", ", free-form text, "Yes"/"No", "Unanswerable"), max over
annotations. Answer F1 is NOT a faithfulness or hallucination measure.
"""
import argparse
import json
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def reference(answer: dict) -> tuple[str, str]:
    """(reference text, answer type) for one QASPER annotation, as in the official evaluator."""
    if answer["unanswerable"]:
        return "Unanswerable", "none"
    if answer["extractive_spans"]:
        return ", ".join(answer["extractive_spans"]), "extractive"
    if answer["free_form_answer"]:
        return answer["free_form_answer"], "abstractive"
    if answer["yes_no"] is not None:
        return ("Yes" if answer["yes_no"] else "No"), "boolean"
    return "", "none"


def answer_f1(prediction: str, annotations: list[dict]) -> float:
    return max(token_f1(prediction, reference(a)[0]) for a in annotations)


def freeze():
    import run_qasper_eval as rq
    frozen = json.loads(rq.FROZEN.read_text())
    rows = {r["id"]: r for r in rq.load_split("test")}
    items = []
    for p in frozen["test"]["papers"]:
        qa = rows[p["paper_id"]]["qas"]
        for i, qid in enumerate(qa["question_id"]):
            anns = qa["answers"][i]["answer"]
            types = Counter(reference(a)[1] for a in anns)
            items.append({"paper_id": p["paper_id"], "question_id": qid, "question": qa["question"][i],
                          "majority_type": types.most_common(1)[0][0], "n_annotations": len(anns)})
    rng = random.Random(20261126)
    pilot = rng.sample(items, 40)
    out = {"source": "38 frozen QASPER test papers (same papers as the retrieval benchmark)",
           "n_questions": len(items), "type_distribution": dict(Counter(i["majority_type"] for i in items)),
           "pilot": {"n": len(pilot), "seed": 20261126, "type_distribution": dict(Counter(i["majority_type"] for i in pilot)),
                     "question_ids": [i["question_id"] for i in pilot]},
           "questions": items}
    (HERE / "answer_set.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out["n_questions"], out["type_distribution"], "pilot", out["pilot"]["type_distribution"])


def plan():
    res = json.loads((HERE / "results_test.json").read_text())["summary"]
    aset = json.loads((HERE / "answer_set.json").read_text())
    ctx = {"fixed": res["fixed|T=256|dense"]["context_tokens@4"], "structure": res["structure|T=256|dense"]["context_tokens@4"]}
    PROMPT, QUESTION, OUT = 180, 25, 150      # tokens: system+instructions, question, max output
    LLAMA_PER_BGE = 1.15                       # rough tokenizer ratio (proxy; verify on first calls)
    JUDGE_CALLS, JUDGE_IN, JUDGE_OUT = 9, 1600, 250  # per sample: faithfulness 2, relevancy 1, ctx precision 4, ctx recall 1, +1 retry margin
    out = {}
    for name, n in (("pilot", aset["pilot"]["n"]), ("full", aset["n_questions"])):
        gen_in = sum(n * (PROMPT + QUESTION + LLAMA_PER_BGE * c) for c in ctx.values())
        samples = 2 * n
        out[name] = {"questions": n, "generation_requests": samples,
                     "generation_input_tokens": round(gen_in), "generation_output_tokens_max": samples * OUT,
                     "ragas_judge_requests_est": samples * JUDGE_CALLS,
                     "ragas_judge_input_tokens_est": samples * JUDGE_CALLS * JUDGE_IN,
                     "ragas_judge_output_tokens_est": samples * JUDGE_CALLS * JUDGE_OUT}
    print(json.dumps({"context_tokens_bge_at_top4": ctx, **out}, indent=1))
    (HERE / "live_eval_plan.json").write_text(json.dumps({"context_tokens_bge_at_top4": ctx, **out}, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    if a.plan:
        plan()
