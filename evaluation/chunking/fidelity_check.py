#!/usr/bin/env python3
"""POST-HOC implementation-fidelity check (run after acceptance; not used for selection):
backend.structure_chunker with its dependency-free token estimator vs the evaluated
configuration C (BGE token counts), on dev and test."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_chunking_eval as rc  # noqa: E402
from backend import structure_chunker  # noqa: E402

rc.CHUNKERS["P"] = lambda path, tok, **_: structure_chunker.chunk_pdf_structured(str(path))
rc.cfg_name_orig = rc.cfg_name
rc.cfg_name = lambda c: "P production structure_chunker (estimator)" if c["family"] == "P" else rc.cfg_name_orig(c)
exp = rc.Experiment()
out = {}
for split, f in (("dev", "dev_evidence.jsonl"), ("test", "test_questions.jsonl")):
    qs = rc.load(f)
    out[split] = [exp.run(c, qs) for c in ({"family": "C", "T": 192, "header": False}, {"family": "P"})]
    for r in out[split]:
        s = r["summary"]
        print(f"{split:4s} {r['name']:44s} R@4 {s['recall@4']:.3f} P@4 {s['precision@4']:.3f} MRR {s['rr@10']:.3f} "
              f"tok {s['context_tokens']:6.1f} chunks {s['n_chunks']} max {s['max_chunk_tokens']} trunc {s['truncated_by_bge_512']}")
(HERE / "results" / "fidelity.json").write_text(json.dumps({"note": __doc__, **out}, indent=1) + "\n")
